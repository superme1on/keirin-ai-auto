import json
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
    TODAY_CSV,
    MODEL_PATH,
    OUTPUT_DIR,
    prepare_features,
    normalize_race_prob,
)


def ensure_ready():
    if not TODAY_CSV.exists():
        print("today_entries.csv not found; generating sample data")
        subprocess.check_call([sys.executable, "src/make_sample_data.py", "--if-missing"])

    if not MODEL_PATH.exists():
        print("model not found; training model")
        subprocess.check_call([sys.executable, "src/train.py"])


def make_trifecta_candidates(race_df: pd.DataFrame, top_k_riders=7, top_n=8):
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

    results = sorted(results, key=lambda x: x["trifecta_prob_approx"], reverse=True)[:top_n]
    return results


def main():
    ensure_dirs()
    ensure_ready()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    fill_values = bundle["fill_values"]

    df = pd.read_csv(TODAY_CSV)
    if "race_id" not in df.columns:
        raise ValueError("today_entries.csv must have race_id column")

    X, _ = prepare_features(df, fill_values)

    raw = model.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw) if calibrator is not None else raw

    pred = df.copy()
    pred["p_raw"] = np.clip(calibrated, 1e-6, 1.0)
    pred = normalize_race_prob(pred, "p_raw", "p_win")
    pred["expected_value_win"] = pred["p_win"] * pd.to_numeric(pred.get("odds_win", np.nan), errors="coerce") - 1
    pred["rank_in_race"] = pred.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)

    sort_cols = ["date", "venue", "race_no", "rank_in_race"]
    pred = pred.sort_values(sort_cols)

    today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    pred_path = OUTPUT_DIR / f"predictions_{today_jst}.csv"
    latest_path = OUTPUT_DIR / "latest_predictions.csv"

    cols = [
        "date", "venue", "race_no", "race_id", "rank_in_race",
        "car_no", "player_id", "style", "score", "odds_win",
        "p_win", "expected_value_win",
    ]
    for c in cols:
        if c not in pred.columns:
            pred[c] = ""

    pred[cols].to_csv(pred_path, index=False)
    pred[cols].to_csv(latest_path, index=False)

    bet_rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        base = g.iloc[0]
        candidates = make_trifecta_candidates(g)
        for i, cand in enumerate(candidates, start=1):
            bet_rows.append({
                "date": base.get("date", ""),
                "venue": base.get("venue", ""),
                "race_no": base.get("race_no", ""),
                "race_id": race_id,
                "candidate_rank": i,
                "buy": cand["buy"],
                "trifecta_prob_approx": cand["trifecta_prob_approx"],
            })

    bets = pd.DataFrame(bet_rows)
    bets_path = OUTPUT_DIR / f"bets_{today_jst}.csv"
    latest_bets_path = OUTPUT_DIR / "latest_bets.csv"
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
        "html": str(html_path),
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pred[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
