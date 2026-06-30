import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import METRICS_PATH, OUTPUT_DIR

GROWTH_LOG_CSV = OUTPUT_DIR / "growth_log.csv"


def latest_prediction_date():
    path = OUTPUT_DIR / "latest_predictions.csv"
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path, usecols=["date"])
        if len(df) and "date" in df.columns:
            return str(df["date"].dropna().max())
    except Exception:
        return ""
    return ""


def latest_purchase_date():
    path = OUTPUT_DIR / "purchase_plan.csv"
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path, usecols=["date"])
        if len(df) and "date" in df.columns:
            return str(df["date"].dropna().max())
    except Exception:
        return ""
    return ""


def read_training_row():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    backtests = metrics.get("backtest", [])
    top1 = next((x for x in backtests if x.get("strategy") == "top_p_win_each_race"), {})

    return {
        "logged_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "kind": "train",
        "target_date": latest_prediction_date(),
        "history_rows": metrics.get("n_rows", np.nan),
        "history_races": metrics.get("n_races", np.nan),
        "auc_binary": metrics.get("auc_binary", np.nan),
        "brier_score_binary": metrics.get("brier_score_binary", np.nan),
        "race_logloss": metrics.get("race_logloss", np.nan),
        "top1_bets": top1.get("bets", np.nan),
        "top1_hits": top1.get("hits", np.nan),
        "top1_hit_rate": top1.get("hit_rate", np.nan),
        "settlement_bets": np.nan,
        "settlement_hits": np.nan,
        "stake_yen": np.nan,
        "return_yen": np.nan,
        "profit_yen": np.nan,
        "roi": np.nan,
        "note": "WINTICKET history training metrics",
    }


def read_settlement_row():
    path = OUTPUT_DIR / "settlement_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pd.read_csv(path)
    row = summary[summary["target"].astype(str).str.contains("AI購入分", na=False)]
    if len(row) == 0:
        row = summary.head(1)
    item = row.iloc[0].to_dict()

    return {
        "logged_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "kind": "settle",
        "target_date": latest_purchase_date() or latest_prediction_date(),
        "history_rows": np.nan,
        "history_races": np.nan,
        "auc_binary": np.nan,
        "brier_score_binary": np.nan,
        "race_logloss": np.nan,
        "top1_bets": np.nan,
        "top1_hits": np.nan,
        "top1_hit_rate": np.nan,
        "settlement_bets": item.get("bets", np.nan),
        "settlement_hits": item.get("hits", np.nan),
        "stake_yen": item.get("stake_yen", np.nan),
        "return_yen": item.get("return_yen", np.nan),
        "profit_yen": item.get("profit_yen", np.nan),
        "roi": item.get("roi", np.nan),
        "note": "AI purchase settlement",
    }


def read_walk_forward_row():
    path = OUTPUT_DIR / "walk_forward_overall.json"
    if not path.exists():
        raise FileNotFoundError(path)
    overall = json.loads(path.read_text(encoding="utf-8"))
    return {
        "logged_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "kind": "walk_forward",
        "target_date": latest_prediction_date(),
        "history_rows": np.nan,
        "history_races": overall.get("test_races", np.nan),
        "auc_binary": overall.get("avg_auc_binary", np.nan),
        "brier_score_binary": np.nan,
        "race_logloss": overall.get("avg_race_logloss", np.nan),
        "top1_bets": overall.get("top1_bets", np.nan),
        "top1_hits": overall.get("top1_hits", np.nan),
        "top1_hit_rate": overall.get("top1_hit_rate", np.nan),
        "settlement_bets": np.nan,
        "settlement_hits": np.nan,
        "stake_yen": np.nan,
        "return_yen": np.nan,
        "profit_yen": np.nan,
        "roi": np.nan,
        "note": "hide-result then predict then reveal walk-forward test",
    }


def read_profit_backtest_row():
    path = OUTPUT_DIR / "profit_backtest_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        "logged_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "kind": "profit_backtest",
        "target_date": latest_prediction_date(),
        "history_rows": np.nan,
        "history_races": summary.get("history_races", np.nan),
        "auc_binary": np.nan,
        "brier_score_binary": np.nan,
        "race_logloss": np.nan,
        "top1_bets": np.nan,
        "top1_hits": np.nan,
        "top1_hit_rate": np.nan,
        "settlement_bets": summary.get("bets", np.nan),
        "settlement_hits": summary.get("hits", np.nan),
        "stake_yen": summary.get("stake_yen", np.nan),
        "return_yen": summary.get("return_yen", np.nan),
        "profit_yen": summary.get("profit_yen", np.nan),
        "roi": summary.get("roi", np.nan),
        "note": "walk-forward trifecta profit backtest with variable stake",
    }


def read_multi_bet_backtest_row():
    path = OUTPUT_DIR / "multi_bet_backtest_overall.json"
    by_type_path = OUTPUT_DIR / "multi_bet_backtest_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    best_note = "multi bet type walk-forward profit backtest"
    if by_type_path.exists():
        by_type = pd.read_csv(by_type_path)
        if len(by_type):
            best = by_type.sort_values("profit_yen", ascending=False).iloc[0]
            best_note = f"best broad bet type: {best.get('bet_label', best.get('bet_type'))}, profit_yen={best.get('profit_yen')}"
    return {
        "logged_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "kind": "multi_bet_backtest",
        "target_date": latest_prediction_date(),
        "history_rows": np.nan,
        "history_races": summary.get("history_races", np.nan),
        "auc_binary": np.nan,
        "brier_score_binary": np.nan,
        "race_logloss": np.nan,
        "top1_bets": np.nan,
        "top1_hits": np.nan,
        "top1_hit_rate": np.nan,
        "settlement_bets": summary.get("overall_bets", np.nan),
        "settlement_hits": summary.get("overall_hits", np.nan),
        "stake_yen": summary.get("overall_stake_yen", np.nan),
        "return_yen": summary.get("overall_return_yen", np.nan),
        "profit_yen": summary.get("overall_profit_yen", np.nan),
        "roi": summary.get("overall_roi", np.nan),
        "note": best_note,
    }


def append_growth_row(kind):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if kind == "train":
        row = read_training_row()
    elif kind == "settle":
        row = read_settlement_row()
    elif kind == "walk_forward":
        row = read_walk_forward_row()
    elif kind == "profit_backtest":
        row = read_profit_backtest_row()
    elif kind == "multi_bet_backtest":
        row = read_multi_bet_backtest_row()
    else:
        raise ValueError(f"unknown kind: {kind}")

    new_df = pd.DataFrame([row])
    if GROWTH_LOG_CSV.exists():
        old = pd.read_csv(GROWTH_LOG_CSV)
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(GROWTH_LOG_CSV, index=False)
    print(f"appended {kind} row to {GROWTH_LOG_CSV}")
    print(new_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=["train", "settle", "walk_forward", "profit_backtest", "multi_bet_backtest"],
        required=True,
    )
    args = parser.parse_args()
    append_growth_row(args.kind)


if __name__ == "__main__":
    main()
