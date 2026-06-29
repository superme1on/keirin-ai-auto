import json
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import (
    ensure_dirs,
    HISTORY_CSV,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLS,
    prepare_features,
    normalize_race_prob,
)


def ensure_history():
    if not HISTORY_CSV.exists():
        print("history.csv not found; generating sample data")
        subprocess.check_call([sys.executable, "src/make_sample_data.py", "--if-missing"])


def safe_auc(y, p):
    try:
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def main():
    ensure_dirs()
    ensure_history()

    df = pd.read_csv(HISTORY_CSV)
    required = {"race_id", "date", "finish_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history.csv missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["target_win"] = (pd.to_numeric(df["finish_pos"], errors="coerce") == 1).astype(int)
    df = df.sort_values(["date", "race_id", "car_no"])

    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 10:
        raise ValueError("not enough dates in history.csv")

    d1 = dates[int(len(dates) * 0.70)]
    d2 = dates[int(len(dates) * 0.85)]

    train_df = df[df["date"] < d1].copy()
    calib_df = df[(df["date"] >= d1) & (df["date"] < d2)].copy()
    test_df = df[df["date"] >= d2].copy()

    if len(train_df) < 100:
        raise ValueError("not enough training rows")

    X_train, fill_values = prepare_features(train_df)
    y_train = train_df["target_win"].astype(int)

    X_calib, _ = prepare_features(calib_df, fill_values)
    y_calib = calib_df["target_win"].astype(int)

    X_test, _ = prepare_features(test_df, fill_values)
    y_test = test_df["target_win"].astype(int)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )

    model.fit(X_train, y_train)

    calibrator = None
    calib_raw = model.predict_proba(X_calib)[:, 1] if len(calib_df) else np.array([])
    if len(calib_raw) > 20 and len(set(y_calib)) == 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_raw, y_calib)

    test_raw = model.predict_proba(X_test)[:, 1]
    test_prob = calibrator.predict(test_raw) if calibrator is not None else test_raw

    pred_df = test_df[["race_id", "finish_pos"]].copy()
    pred_df["p_raw"] = test_prob
    pred_df = normalize_race_prob(pred_df)

    winner_probs = pred_df[pred_df["finish_pos"] == 1]["p_win"].clip(1e-9, 1.0)
    race_logloss = float(-np.log(winner_probs).mean()) if len(winner_probs) else None

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "model": "HistGradientBoostingClassifier + race probability normalization",
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
        "n_races": int(df["race_id"].nunique()),
        "features": FEATURE_COLS,
        "brier_score_binary": float(brier_score_loss(y_test, test_prob)) if len(test_df) else None,
        "auc_binary": safe_auc(y_test, test_prob),
        "race_logloss": race_logloss,
    }

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "fill_values": fill_values,
        "features": FEATURE_COLS,
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
