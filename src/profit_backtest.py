import argparse
import json
from itertools import permutations
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    HISTORY_CSV,
    HISTORY_TRIFECTA_ODDS_CSV,
    OUTPUT_DIR,
    FEATURE_COLS,
    ensure_dirs,
    normalize_race_prob,
    prepare_features,
)


def load_history_odds():
    columns = ["race_id", "buy", "trifecta_odds", "is_actual", "actual_trifecta"]
    if not HISTORY_TRIFECTA_ODDS_CSV.exists() or HISTORY_TRIFECTA_ODDS_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        odds = pd.read_csv(HISTORY_TRIFECTA_ODDS_CSV, dtype={"race_id": str, "buy": str})
    except EmptyDataError:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in odds.columns:
            odds[col] = np.nan
    odds["trifecta_odds"] = pd.to_numeric(odds["trifecta_odds"], errors="coerce")
    return odds


def make_candidates(race_df, top_k=5):
    riders = race_df.sort_values("p_win", ascending=False).head(top_k)
    rows = riders[["car_no", "p_win"]].to_dict("records")
    out = []
    for a, b, c in permutations(rows, 3):
        p1 = float(a["p_win"])
        p2 = float(b["p_win"]) / max(1.0 - p1, 1e-9)
        p3 = float(c["p_win"]) / max(1.0 - p1 - float(b["p_win"]), 1e-9)
        prob = max(0.0, min(1.0, p1 * p2 * p3))
        out.append({"buy": f'{int(a["car_no"])}-{int(b["car_no"])}-{int(c["car_no"])}', "trifecta_prob_approx": prob})
    return pd.DataFrame(out, columns=["buy", "trifecta_prob_approx"])


def stake_from_edge(expected_profit_yen, base_stake=100, max_stake=500):
    if expected_profit_yen < 100:
        return base_stake
    extra_units = int(min((expected_profit_yen // 300), (max_stake - base_stake) // 100))
    return int(base_stake + extra_units * 100)


def train_model(train_df):
    X, fill_values = prepare_features(train_df)
    y = train_df["target_win"].astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=42,
    )
    model.fit(X, y)
    return model, fill_values


def run_profit_backtest(
    min_train_dates=10,
    min_prob=0.015,
    min_expected_profit=100,
    max_odds=300,
    max_bets_per_race=2,
    max_bets_per_day=40,
    base_stake=100,
    max_stake=500,
):
    ensure_dirs()
    hist = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    odds = load_history_odds()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["target_win"] = (pd.to_numeric(hist["finish_pos"], errors="coerce") == 1).astype(int)
    odds_race_count = int(odds["race_id"].nunique()) if len(odds) else 0

    dates = sorted(hist["date"].dropna().unique())
    all_bets = []
    model = None
    fill_values = None

    for i, test_date in enumerate(dates[min_train_dates:], start=min_train_dates):
        train_df = hist[hist["date"] < test_date].copy()
        test_df = hist[hist["date"] == test_date].copy()
        if len(train_df) < 100 or len(test_df) == 0:
            continue

        model, fill_values = train_model(train_df)
        X_test, _ = prepare_features(test_df, fill_values)
        pred = test_df.copy()
        pred["p_raw"] = model.predict_proba(X_test)[:, 1]
        pred = normalize_race_prob(pred)

        day_bets = []
        for race_id, race_df in pred.groupby("race_id"):
            cand = make_candidates(race_df, top_k=5)
            race_odds = odds[odds["race_id"] == str(race_id)][["race_id", "buy", "trifecta_odds", "is_actual", "actual_trifecta"]]
            merged = cand.merge(race_odds, on="buy", how="inner")
            if len(merged) == 0:
                continue
            base = race_df.iloc[0]
            merged["race_id"] = str(race_id)
            merged["date"] = base["date"].strftime("%Y-%m-%d")
            merged["venue"] = base.get("venue", "")
            merged["race_no"] = base.get("race_no", "")
            merged["expected_profit_100yen"] = (merged["trifecta_prob_approx"] * merged["trifecta_odds"] - 1) * 100
            selected = merged[
                (merged["trifecta_prob_approx"] >= min_prob)
                & (merged["expected_profit_100yen"] >= min_expected_profit)
                & (merged["trifecta_odds"] <= max_odds)
            ].sort_values("expected_profit_100yen", ascending=False).head(max_bets_per_race)
            day_bets.append(selected)

        if day_bets:
            day = pd.concat(day_bets, ignore_index=True).sort_values("expected_profit_100yen", ascending=False).head(max_bets_per_day)
            all_bets.append(day)

    if all_bets:
        bets = pd.concat(all_bets, ignore_index=True)
    else:
        bets = pd.DataFrame(columns=["date", "venue", "race_no", "race_id", "buy", "trifecta_prob_approx", "trifecta_odds"])

    if len(bets):
        bets["stake_yen"] = bets["expected_profit_100yen"].map(lambda x: stake_from_edge(x, base_stake, max_stake))
        bets["is_hit"] = bets["is_actual"].astype(bool)
        bets["return_yen"] = np.where(bets["is_hit"], (bets["stake_yen"] * bets["trifecta_odds"]).round(0), 0)
        bets["profit_yen"] = bets["return_yen"] - bets["stake_yen"]
    else:
        bets["stake_yen"] = []
        bets["is_hit"] = []
        bets["return_yen"] = []
        bets["profit_yen"] = []

    total_stake = int(pd.to_numeric(bets.get("stake_yen", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_return = int(pd.to_numeric(bets.get("return_yen", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_profit = int(pd.to_numeric(bets.get("profit_yen", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    summary = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "mode": "walk-forward profit backtest",
        "features": FEATURE_COLS,
        "bets": int(len(bets)),
        "hits": int(bets["is_hit"].sum()) if len(bets) else 0,
        "hit_rate": float(bets["is_hit"].mean()) if len(bets) else None,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_profit,
        "roi": float(total_profit / total_stake) if total_stake else None,
        "history_races": int(hist["race_id"].nunique()),
        "odds_races": odds_race_count,
        "note": "" if odds_race_count else "過去三連単オッズが未取得のため、損益検証の買い目は作成されませんでした。",
        "params": {
            "min_train_dates": min_train_dates,
            "min_prob": min_prob,
            "min_expected_profit": min_expected_profit,
            "max_odds": max_odds,
            "max_bets_per_race": max_bets_per_race,
            "max_bets_per_day": max_bets_per_day,
            "base_stake": base_stake,
            "max_stake": max_stake,
        },
    }

    bets.to_csv(OUTPUT_DIR / "profit_backtest_bets.csv", index=False)
    (OUTPUT_DIR / "profit_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).drop(columns=["params", "features"]).to_csv(OUTPUT_DIR / "profit_backtest_summary.csv", index=False)
    write_japanese_report(summary, bets)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def yen(value):
    return f"{int(value):,}円"


def write_japanese_report(summary, bets):
    hit_rate = "-" if summary["hit_rate"] is None else f"{summary['hit_rate']:.1%}"
    roi = "-" if summary["roi"] is None else f"{summary['roi']:.1%}"
    lines = [
        "# 損益バックテスト",
        "",
        f"- 作成日時: {summary['created_at_jst']}",
        f"- 検証レース数: {summary['history_races']:,}",
        f"- オッズ取得済みレース数: {summary['odds_races']:,}",
        f"- 買い目数: {summary['bets']:,}",
        f"- 的中数: {summary['hits']:,}",
        f"- 的中率: {hit_rate}",
        f"- 購入金額: {yen(summary['stake_yen'])}",
        f"- 払戻金額: {yen(summary['return_yen'])}",
        f"- 損益: {yen(summary['profit_yen'])}",
        f"- 回収率: {roi}",
    ]
    if summary.get("note"):
        lines += ["", f"注意: {summary['note']}"]
    if len(bets):
        top = bets.sort_values("profit_yen", ascending=False).head(10)
        lines += ["", "## 利益上位の買い目", "", "|日付|場|R|買い目|購入|オッズ|結果|損益|", "|---|---|---:|---|---:|---:|---|---:|"]
        for _, row in top.iterrows():
            result = "的中" if row["is_hit"] else f"外れ({row.get('actual_trifecta', '')})"
            lines.append(
                f"|{row['date']}|{row['venue']}|{int(row['race_no'])}|{row['buy']}|"
                f"{int(row['stake_yen'])}|{float(row['trifecta_odds']):.1f}|{result}|{int(row['profit_yen'])}|"
            )
    (OUTPUT_DIR / "profit_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-dates", type=int, default=10)
    parser.add_argument("--min-prob", type=float, default=0.015)
    parser.add_argument("--min-expected-profit", type=float, default=100)
    parser.add_argument("--max-odds", type=float, default=300)
    parser.add_argument("--max-bets-per-race", type=int, default=2)
    parser.add_argument("--max-bets-per-day", type=int, default=40)
    parser.add_argument("--base-stake", type=int, default=100)
    parser.add_argument("--max-stake", type=int, default=500)
    args = parser.parse_args()
    run_profit_backtest(
        args.min_train_dates,
        args.min_prob,
        args.min_expected_profit,
        args.max_odds,
        args.max_bets_per_race,
        args.max_bets_per_day,
        args.base_stake,
        args.max_stake,
    )


if __name__ == "__main__":
    main()
