import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import permutations
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from common import (
    ensure_dirs,
    HISTORY_CSV,
    RAW_DIR,
    TODAY_CSV,
    TODAY_ODDS_CSV,
    MODEL_PATH,
    OUTPUT_DIR,
    add_player_prior_features,
    prepare_features,
    normalize_race_prob,
)
from multi_bet_backtest import BET_LABELS, make_candidates as make_multi_bet_candidates

TRIFECTA_ODDS_CSV = RAW_DIR / "today_trifecta_odds.csv"
PROFIT_GATE_PATH = OUTPUT_DIR / "external_holdout_overall.json"
DEFAULT_BET_CONFIGS = {
    "exacta": {"min_prob": 0.05, "min_ev": 800, "max_odds": 200, "max_per_race": 2},
    "quinella": {"min_prob": 0.05, "min_ev": 1200, "max_odds": 300, "max_per_race": 2},
    "quinella_place": {"min_prob": 0.10, "min_ev": 300, "max_odds": 100, "max_per_race": 2},
    "trio": {"min_prob": 0.07, "min_ev": 800, "max_odds": 300, "max_per_race": 2},
    "trifecta": {"min_prob": 0.07, "min_ev": 800, "max_odds": 300, "max_per_race": 2},
}


def get_stake_yen():
    raw = os.getenv("BET_STAKE_YEN", "100").strip()
    try:
        stake = int(raw)
    except ValueError:
        stake = 100
    return max(stake, 0)


def get_float_env(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def get_int_env(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def load_profit_gate():
    if not PROFIT_GATE_PATH.exists():
        return {"target_passed": False, "reason": "external holdout has not been completed"}
    try:
        result = json.loads(PROFIT_GATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"target_passed": False, "reason": "external holdout result is unreadable"}
    if not bool(result.get("target_passed", False)):
        return {
            "target_passed": False,
            "reason": "external holdout ROI target was not met",
            "roi": result.get("roi"),
            "bets": result.get("bets"),
        }
    return {"target_passed": True, "reason": "external holdout gate passed", **result}


def stake_from_edge(expected_profit_100yen, base_stake=100, max_stake=500):
    if pd.isna(expected_profit_100yen):
        return base_stake
    if expected_profit_100yen < 100:
        return base_stake
    extra_units = int(min((expected_profit_100yen // 300), (max_stake - base_stake) // 100))
    return int(base_stake + extra_units * 100)


def ensure_ready():
    if not TODAY_CSV.exists():
        print("today_entries.csv not found; generating sample data")
        subprocess.check_call([sys.executable, "src/make_sample_data.py", "--if-missing"])

    if not MODEL_PATH.exists():
        print("model not found; training model")
        subprocess.check_call([sys.executable, "src/train.py"])


def make_trifecta_candidates(race_df: pd.DataFrame, top_k_riders=5):
    riders = race_df.sort_values("p_win", ascending=False).head(top_k_riders)
    rows = riders[["car_no", "p_win"]].to_dict("records")
    results = []

    for a, b, c in permutations(rows, 3):
        p1 = float(a["p_win"])
        denom2 = max(1.0 - p1, 1e-9)
        p2 = float(b["p_win"]) / denom2
        denom3 = max(1.0 - p1 - float(b["p_win"]), 1e-9)
        p3 = float(c["p_win"]) / denom3
        prob = max(0.0, min(1.0, p1 * p2 * p3))

        results.append({
            "buy": f'{int(a["car_no"])}-{int(b["car_no"])}-{int(c["car_no"])}',
            "trifecta_prob_approx": prob,
        })

    results = sorted(results, key=lambda x: x["trifecta_prob_approx"], reverse=True)
    return results


def load_trifecta_odds():
    if not TRIFECTA_ODDS_CSV.exists():
        return pd.DataFrame(columns=["race_id", "buy", "trifecta_odds"])
    odds = pd.read_csv(TRIFECTA_ODDS_CSV, dtype={"race_id": str, "buy": str})
    if "race_id" not in odds.columns or "buy" not in odds.columns:
        return pd.DataFrame(columns=["race_id", "buy", "trifecta_odds"])
    odds["race_id"] = odds["race_id"].astype(str)
    odds["buy"] = odds["buy"].astype(str)
    odds["trifecta_odds"] = pd.to_numeric(odds.get("trifecta_odds", np.nan), errors="coerce")
    return odds[["race_id", "buy", "trifecta_odds"]]


def load_today_odds():
    if TODAY_ODDS_CSV.exists():
        odds = pd.read_csv(TODAY_ODDS_CSV, dtype={"race_id": str, "buy": str, "bet_type": str})
        required = ["race_id", "bet_type", "buy", "odds_used"]
        if all(c in odds.columns for c in required):
            odds["race_id"] = odds["race_id"].astype(str)
            odds["buy"] = odds["buy"].astype(str)
            odds["bet_type"] = odds["bet_type"].astype(str)
            odds["odds_used"] = pd.to_numeric(odds["odds_used"], errors="coerce")
            return odds[required]

    trifecta = load_trifecta_odds()
    if len(trifecta):
        trifecta = trifecta.rename(columns={"trifecta_odds": "odds_used"})
        trifecta["bet_type"] = "trifecta"
        return trifecta[["race_id", "bet_type", "buy", "odds_used"]]
    return pd.DataFrame(columns=["race_id", "bet_type", "buy", "odds_used"])


def add_today_prior_features(df):
    if not HISTORY_CSV.exists():
        return df
    hist = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    df = df.copy()
    hist["_today_row_id"] = np.nan
    df["_today_row_id"] = np.arange(len(df))
    if "finish_pos" not in df.columns:
        df["finish_pos"] = np.nan
    combined = pd.concat([hist, df], ignore_index=True, sort=False)
    combined = add_player_prior_features(combined)
    today = combined[combined["_today_row_id"].notna()].copy()
    today = today.sort_values("_today_row_id", kind="mergesort")
    return today.drop(columns=["_today_row_id"], errors="ignore")


def main():
    ensure_dirs()
    ensure_ready()
    base_stake_yen = get_int_env("BET_BASE_STAKE_YEN", get_stake_yen())
    max_stake_yen = get_int_env("BET_MAX_STAKE_YEN", 500)
    profit_gate = load_profit_gate()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    fill_values = bundle["fill_values"]

    df = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    if "race_id" not in df.columns:
        raise ValueError("today_entries.csv must have race_id column")
    df = add_today_prior_features(df)

    X, _ = prepare_features(df, fill_values)

    raw = model.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw) if calibrator is not None else raw

    pred = df.copy()
    pred["p_raw"] = np.clip(calibrated, 1e-6, 1.0)
    pred = normalize_race_prob(pred, "p_raw", "p_win")
    pred["expected_value_win"] = pred["p_win"] * pd.to_numeric(pred.get("odds_win", np.nan), errors="coerce") - 1
    pred["stake_yen"] = base_stake_yen
    pred["win_return_yen"] = (base_stake_yen * pd.to_numeric(pred.get("odds_win", np.nan), errors="coerce")).round(0)
    pred["win_profit_yen"] = pred["win_return_yen"] - base_stake_yen
    pred["loss_amount_yen"] = base_stake_yen
    pred["expected_profit_yen"] = (base_stake_yen * pred["expected_value_win"]).round(0)
    pred["rank_in_race"] = pred.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)

    sort_cols = ["date", "venue", "race_no", "rank_in_race"]
    pred = pred.sort_values(sort_cols)

    today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    pred_path = OUTPUT_DIR / f"predictions_{today_jst}.csv"
    latest_path = OUTPUT_DIR / "latest_predictions.csv"

    cols = [
        "date", "venue", "race_no", "race_id", "rank_in_race",
        "car_no", "player_id", "style", "score", "odds_win",
        "p_win", "expected_value_win", "stake_yen", "win_return_yen",
        "win_profit_yen", "loss_amount_yen", "expected_profit_yen",
    ]
    for c in cols:
        if c not in pred.columns:
            pred[c] = ""

    pred[cols].to_csv(pred_path, index=False)
    pred[cols].to_csv(latest_path, index=False)

    bet_rows = []
    today_odds = load_today_odds()
    for race_id, g in pred.groupby("race_id", sort=False):
        base = g.iloc[0]
        candidates = make_multi_bet_candidates(g, top_k=5)
        race_odds = today_odds[today_odds["race_id"].eq(str(race_id))]
        if len(candidates) == 0 or len(race_odds) == 0:
            continue
        merged = candidates.merge(race_odds, on=["bet_type", "buy"], how="inner")
        if len(merged) == 0:
            continue
        merged = merged.sort_values(["bet_type", "prob"], ascending=[True, False])
        merged["candidate_rank"] = merged.groupby("bet_type")["prob"].rank(ascending=False, method="first").astype(int)
        for _, cand in merged.iterrows():
            expected_value = cand["prob"] * cand["odds_used"] - 1 if pd.notna(cand["odds_used"]) else np.nan
            bet_rows.append({
                "date": base.get("date", ""),
                "venue": base.get("venue", ""),
                "race_no": base.get("race_no", ""),
                "race_id": race_id,
                "bet_type": cand["bet_type"],
                "bet_label": BET_LABELS.get(cand["bet_type"], cand["bet_type"]),
                "candidate_rank": int(cand["candidate_rank"]),
                "buy": cand["buy"],
                "prob": cand["prob"],
                "odds_used": cand["odds_used"],
                "expected_profit_100yen": round(100 * expected_value) if pd.notna(expected_value) else np.nan,
            })

    candidates = pd.DataFrame(bet_rows)
    if len(candidates):
        candidates["is_selected"] = False
        for bet_type, config in DEFAULT_BET_CONFIGS.items():
            mask = candidates["bet_type"].eq(bet_type)
            candidates.loc[mask, "is_selected"] = (
                (pd.to_numeric(candidates.loc[mask, "prob"], errors="coerce") >= config["min_prob"])
                & (pd.to_numeric(candidates.loc[mask, "expected_profit_100yen"], errors="coerce") >= config["min_ev"])
                & (pd.to_numeric(candidates.loc[mask, "odds_used"], errors="coerce") <= config["max_odds"])
            )
        selected_parts = []
        for (_, bet_type), g in candidates[candidates["is_selected"]].groupby(["race_id", "bet_type"], sort=False):
            max_per_race = DEFAULT_BET_CONFIGS.get(bet_type, {}).get("max_per_race", 1)
            selected_parts.append(g.sort_values("expected_profit_100yen", ascending=False).head(max_per_race))
        bets = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.head(0).copy()
        if len(bets):
            bets["stake_yen"] = bets["expected_profit_100yen"].map(
                lambda x: stake_from_edge(x, base_stake_yen, max_stake_yen)
            )
            bets["return_if_hit_yen"] = (bets["stake_yen"] * pd.to_numeric(bets["odds_used"], errors="coerce")).round(0)
            bets["profit_if_hit_yen"] = bets["return_if_hit_yen"] - bets["stake_yen"]
            bets["loss_amount_yen"] = bets["stake_yen"]
            bets["expected_profit_yen"] = (bets["stake_yen"] * pd.to_numeric(bets["expected_profit_100yen"], errors="coerce") / 100).round(0)
    else:
        candidates = pd.DataFrame(columns=["date", "venue", "race_no", "race_id", "bet_type", "bet_label", "candidate_rank", "buy"])
        bets = candidates.copy()

    shadow_bets = bets.copy()
    if len(shadow_bets):
        shadow_bets["purchase_authorized"] = bool(profit_gate["target_passed"])
        shadow_bets["authorization_reason"] = profit_gate["reason"]
    shadow_path = OUTPUT_DIR / f"shadow_bets_{today_jst}.csv"
    latest_shadow_path = OUTPUT_DIR / "latest_shadow_bets.csv"
    shadow_bets.to_csv(shadow_path, index=False)
    shadow_bets.to_csv(latest_shadow_path, index=False)
    if not profit_gate["target_passed"]:
        bets = bets.head(0).copy()

    bets_path = OUTPUT_DIR / f"bets_{today_jst}.csv"
    latest_bets_path = OUTPUT_DIR / "latest_bets.csv"
    latest_candidates_path = OUTPUT_DIR / "latest_bet_candidates.csv"
    candidates.to_csv(latest_candidates_path, index=False)
    bets.to_csv(bets_path, index=False)
    bets.to_csv(latest_bets_path, index=False)

    html_path = OUTPUT_DIR / "index.html"
    top_table = pred[cols].head(100).copy()
    for col in ["p_win", "expected_value_win"]:
        top_table[col] = pd.to_numeric(top_table[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    html = f"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>Keirin AI Predictions</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Keirin AI Predictions</h1>
  <p>Generated at JST: {datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")}</p>
  <h2>Top predictions</h2>
  {top_table.to_html(index=False, escape=False)}
  <h2>Trifecta candidates</h2>
  {bets.head(100).to_html(index=False, escape=False) if len(bets) else "<p>No bets</p>"}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    summary = {
        "created": str(latest_path),
        "bets": str(latest_bets_path),
        "candidates": str(latest_candidates_path),
        "shadow_bets": str(latest_shadow_path),
        "html": str(html_path),
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
        "base_stake_yen": base_stake_yen,
        "max_stake_yen": max_stake_yen,
        "selected_bets": int(len(bets)),
        "shadow_selected_bets": int(len(shadow_bets)),
        "profit_gate": profit_gate,
        "bet_filter": {
            "configs": DEFAULT_BET_CONFIGS,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pred[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
