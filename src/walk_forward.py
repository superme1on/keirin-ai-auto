import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from common import HISTORY_CSV, OUTPUT_DIR, FEATURE_COLS, ensure_dirs, normalize_race_prob, prepare_features


def safe_auc(y, p):
    try:
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def train_model(train_df):
    X_train, fill_values = prepare_features(train_df)
    y_train = train_df["target_win"].astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, fill_values


def run_walk_forward(min_train_dates=10, retrain_every_days=1):
    ensure_dirs()
    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    df["date"] = pd.to_datetime(df["date"])
    df["target_win"] = (pd.to_numeric(df["finish_pos"], errors="coerce") == 1).astype(int)
    df = df.sort_values(["date", "race_id", "car_no"])

    dates = sorted(df["date"].dropna().unique())
    if len(dates) <= min_train_dates:
        raise ValueError(f"not enough dates for walk-forward: {len(dates)}")

    all_preds = []
    day_rows = []
    model = None
    fill_values = None
    last_train_index = None

    for i, test_date in enumerate(dates[min_train_dates:], start=min_train_dates):
        train_df = df[df["date"] < test_date].copy()
        test_df = df[df["date"] == test_date].copy()
        if len(train_df) < 100 or len(test_df) == 0:
            continue

        if model is None or last_train_index is None or (i - last_train_index) >= retrain_every_days:
            model, fill_values = train_model(train_df)
            last_train_index = i

        X_test, _ = prepare_features(test_df, fill_values)
        raw = model.predict_proba(X_test)[:, 1]
        pred = test_df.copy()
        pred["p_raw"] = raw
        pred = normalize_race_prob(pred, "p_raw", "p_win")
        pred["rank_in_race"] = pred.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
        pred["is_hit"] = pred["finish_pos"].eq(1)
        all_preds.append(pred)

        top1 = pred[pred["rank_in_race"] == 1].copy()
        winner_probs = pred[pred["finish_pos"] == 1]["p_win"].clip(1e-9, 1.0)
        day_rows.append(
            {
                "date": pd.Timestamp(test_date).strftime("%Y-%m-%d"),
                "train_rows": int(len(train_df)),
                "train_races": int(train_df["race_id"].nunique()),
                "test_rows": int(len(test_df)),
                "test_races": int(test_df["race_id"].nunique()),
                "top1_bets": int(len(top1)),
                "top1_hits": int(top1["is_hit"].sum()),
                "top1_hit_rate": float(top1["is_hit"].mean()) if len(top1) else None,
                "auc_binary": safe_auc(test_df["target_win"].astype(int), raw),
                "race_logloss": float(-np.log(winner_probs).mean()) if len(winner_probs) else None,
            }
        )

    if not all_preds:
        raise ValueError("walk-forward produced no predictions")

    pred_df = pd.concat(all_preds, ignore_index=True)
    summary = pd.DataFrame(day_rows)
    total_top1_bets = int(summary["top1_bets"].sum())
    total_top1_hits = int(summary["top1_hits"].sum())
    overall = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "features": FEATURE_COLS,
        "days": int(len(summary)),
        "test_races": int(summary["test_races"].sum()),
        "top1_bets": total_top1_bets,
        "top1_hits": total_top1_hits,
        "top1_hit_rate": float(total_top1_hits / total_top1_bets) if total_top1_bets else None,
        "avg_auc_binary": float(summary["auc_binary"].dropna().mean()) if summary["auc_binary"].notna().any() else None,
        "avg_race_logloss": float(summary["race_logloss"].dropna().mean()) if summary["race_logloss"].notna().any() else None,
    }

    pred_cols = [
        "date", "venue", "race_no", "race_id", "rank_in_race", "car_no", "player_id",
        "player_name", "age", "style", "score", "win_rate", "place2_rate", "place3_rate",
        "race_class", "race_type", "distance", "p_win", "finish_pos", "is_hit",
    ]
    for col in pred_cols:
        if col not in pred_df.columns:
            pred_df[col] = ""

    pred_df[pred_cols].sort_values(["date", "venue", "race_no", "rank_in_race"]).to_csv(
        OUTPUT_DIR / "walk_forward_predictions.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "walk_forward_summary.csv", index=False)
    (OUTPUT_DIR / "walk_forward_overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-dates", type=int, default=10)
    parser.add_argument("--retrain-every-days", type=int, default=1)
    args = parser.parse_args()
    run_walk_forward(args.min_train_dates, args.retrain_every_days)


if __name__ == "__main__":
    main()
