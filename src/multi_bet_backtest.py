import argparse
import json
from datetime import datetime
from itertools import combinations, permutations
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    FEATURE_COLS,
    HISTORY_CSV,
    HISTORY_ODDS_CSV,
    OUTPUT_DIR,
    add_player_prior_features,
    ensure_dirs,
    normalize_race_prob,
    prepare_features,
)


BET_LABELS = {
    "exacta": "2車単",
    "quinella": "2車複",
    "quinella_place": "ワイド",
    "trio": "3連複",
    "trifecta": "3連単",
    "bracket_exacta": "枠単",
    "bracket_quinella": "枠複",
}


def load_history_odds():
    columns = ["race_id", "bet_type", "buy", "odds_used", "is_actual", "actual_buy"]
    if not HISTORY_ODDS_CSV.exists() or HISTORY_ODDS_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        odds = pd.read_csv(HISTORY_ODDS_CSV, dtype={"race_id": str, "buy": str, "bet_type": str})
    except EmptyDataError:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in odds.columns:
            odds[col] = np.nan
    odds["odds_used"] = pd.to_numeric(odds["odds_used"], errors="coerce")
    return odds


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


def pl2(a, b):
    p1 = float(a["p_win"])
    p2 = float(b["p_win"]) / max(1.0 - p1, 1e-9)
    return max(0.0, min(1.0, p1 * p2))


def pl3(a, b, c):
    p1 = float(a["p_win"])
    p2 = float(b["p_win"]) / max(1.0 - p1, 1e-9)
    p3 = float(c["p_win"]) / max(1.0 - p1 - float(b["p_win"]), 1e-9)
    return max(0.0, min(1.0, p1 * p2 * p3))


def make_candidates(race_df, top_k=5):
    riders = race_df.sort_values("p_win", ascending=False).head(top_k)
    rows = riders[["car_no", "p_win"]].to_dict("records")
    cars = [int(r["car_no"]) for r in rows]
    by_car = {int(r["car_no"]): r for r in rows}
    out = []

    for a in rows:
        for b in rows:
            if int(a["car_no"]) == int(b["car_no"]):
                continue
            out.append({"bet_type": "exacta", "buy": f'{int(a["car_no"])}-{int(b["car_no"])}', "prob": pl2(a, b)})

    for a, b in combinations(cars, 2):
        prob = pl2(by_car[a], by_car[b]) + pl2(by_car[b], by_car[a])
        out.append({"bet_type": "quinella", "buy": f"{a}-{b}", "prob": prob})

    for a, b, c in permutations(rows, 3):
        out.append({
            "bet_type": "trifecta",
            "buy": f'{int(a["car_no"])}-{int(b["car_no"])}-{int(c["car_no"])}',
            "prob": pl3(a, b, c),
        })

    for combo in combinations(cars, 3):
        prob = 0.0
        for perm in permutations(combo, 3):
            prob += pl3(by_car[perm[0]], by_car[perm[1]], by_car[perm[2]])
        out.append({"bet_type": "trio", "buy": "-".join(str(x) for x in combo), "prob": prob})

    for pair in combinations(cars, 2):
        prob = 0.0
        for third in cars:
            if third in pair:
                continue
            for perm in permutations([pair[0], pair[1], third], 3):
                prob += pl3(by_car[perm[0]], by_car[perm[1]], by_car[perm[2]])
        out.append({"bet_type": "quinella_place", "buy": "-".join(str(x) for x in pair), "prob": prob})

    return pd.DataFrame(out, columns=["bet_type", "buy", "prob"])


def stake_from_edge(expected_profit_100yen, base_stake=100, max_stake=500):
    if pd.isna(expected_profit_100yen) or expected_profit_100yen < 100:
        return base_stake
    extra_units = int(min((expected_profit_100yen // 300), (max_stake - base_stake) // 100))
    return int(base_stake + extra_units * 100)


def summarize_bets(bets):
    if len(bets) == 0:
        return {
            "bets": 0,
            "hits": 0,
            "hit_rate": None,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": None,
        }
    stake = int(pd.to_numeric(bets["stake_yen"], errors="coerce").fillna(0).sum())
    returns = int(pd.to_numeric(bets["return_yen"], errors="coerce").fillna(0).sum())
    profit = int(pd.to_numeric(bets["profit_yen"], errors="coerce").fillna(0).sum())
    return {
        "bets": int(len(bets)),
        "hits": int(bets["is_hit"].sum()),
        "hit_rate": float(bets["is_hit"].mean()),
        "stake_yen": stake,
        "return_yen": returns,
        "profit_yen": profit,
        "roi": float(profit / stake) if stake else None,
    }


def run_multi_bet_backtest(
    min_train_dates=30,
    retrain_every_days=7,
    top_k=5,
    min_prob=0.02,
    min_expected_profit=100,
    max_odds=300,
    max_bets_per_race_type=2,
    max_bets_per_day_type=40,
    base_stake=100,
    max_stake=500,
):
    ensure_dirs()
    hist = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    odds = load_history_odds()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["target_win"] = (pd.to_numeric(hist["finish_pos"], errors="coerce") == 1).astype(int)
    hist = add_player_prior_features(hist)
    odds = odds[odds["bet_type"].isin(BET_LABELS)].copy()
    odds = odds[pd.to_numeric(odds["odds_used"], errors="coerce").gt(0)].copy()
    odds_cols = ["race_id", "bet_type", "buy", "odds_used", "is_actual", "actual_buy"]
    odds_by_race = {
        str(race_id): race_odds[odds_cols].copy()
        for race_id, race_odds in odds.groupby("race_id", sort=False)
    }

    dates = sorted(hist["date"].dropna().unique())
    all_bets = []
    model = None
    fill_values = None
    last_train_index = None

    for i, test_date in enumerate(dates[min_train_dates:], start=min_train_dates):
        train_df = hist[hist["date"] < test_date].copy()
        test_df = hist[hist["date"] == test_date].copy()
        if len(train_df) < 100 or len(test_df) == 0:
            continue
        if model is None or last_train_index is None or (i - last_train_index) >= retrain_every_days:
            model, fill_values = train_model(train_df)
            last_train_index = i

        X_test, _ = prepare_features(test_df, fill_values)
        pred = test_df.copy()
        pred["p_raw"] = model.predict_proba(X_test)[:, 1]
        pred = normalize_race_prob(pred)

        day_selected = []
        for race_id, race_df in pred.groupby("race_id"):
            cand = make_candidates(race_df, top_k=top_k)
            race_odds = odds_by_race.get(str(race_id))
            if race_odds is None:
                continue
            merged = cand.merge(race_odds, on=["bet_type", "buy"], how="inner")
            if len(merged) == 0:
                continue
            base = race_df.iloc[0]
            merged["race_id"] = str(race_id)
            merged["date"] = base["date"].strftime("%Y-%m-%d")
            merged["venue"] = base.get("venue", "")
            merged["race_no"] = base.get("race_no", "")
            merged["expected_profit_100yen"] = (merged["prob"] * merged["odds_used"] - 1) * 100
            selected = merged[
                (merged["prob"] >= min_prob)
                & (merged["expected_profit_100yen"] >= min_expected_profit)
                & (merged["odds_used"] <= max_odds)
            ]
            if len(selected):
                selected = (
                    selected.sort_values("expected_profit_100yen", ascending=False)
                    .groupby("bet_type", as_index=False, sort=False)
                    .head(max_bets_per_race_type)
                )
                day_selected.append(selected)

        if day_selected:
            day = pd.concat(day_selected, ignore_index=True)
            day = (
                day.sort_values("expected_profit_100yen", ascending=False)
                .groupby("bet_type", as_index=False, sort=False)
                .head(max_bets_per_day_type)
            )
            all_bets.append(day)

    if all_bets:
        bets = pd.concat(all_bets, ignore_index=True)
    else:
        bets = pd.DataFrame(columns=["date", "venue", "race_no", "race_id", "bet_type", "buy", "prob", "odds_used"])

    if len(bets):
        bets["stake_yen"] = bets["expected_profit_100yen"].map(lambda x: stake_from_edge(x, base_stake, max_stake))
        bets["is_hit"] = bets["is_actual"].astype(bool)
        bets["return_yen"] = np.where(bets["is_hit"], (bets["stake_yen"] * bets["odds_used"]).round(0), 0)
        bets["profit_yen"] = bets["return_yen"] - bets["stake_yen"]
        bets["bet_label"] = bets["bet_type"].map(BET_LABELS)

    summary_rows = []
    for bet_type in BET_LABELS:
        d = bets[bets["bet_type"].eq(bet_type)] if len(bets) else bets
        row = {
            "bet_type": bet_type,
            "bet_label": BET_LABELS[bet_type],
            **summarize_bets(d),
        }
        summary_rows.append(row)
    by_type = pd.DataFrame(summary_rows)
    overall = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "mode": "multi bet type walk-forward profit backtest",
        "features": FEATURE_COLS,
        "history_races": int(hist["race_id"].nunique()),
        "odds_races": int(odds["race_id"].nunique()) if len(odds) else 0,
        "params": {
            "min_train_dates": min_train_dates,
            "retrain_every_days": retrain_every_days,
            "top_k": top_k,
            "min_prob": min_prob,
            "min_expected_profit": min_expected_profit,
            "max_odds": max_odds,
            "max_bets_per_race_type": max_bets_per_race_type,
            "max_bets_per_day_type": max_bets_per_day_type,
            "base_stake": base_stake,
            "max_stake": max_stake,
            "wide_odds_policy": "ワイドは下限オッズを使用",
        },
    }
    overall.update({f"overall_{k}": v for k, v in summarize_bets(bets).items()})

    bets.to_csv(OUTPUT_DIR / "multi_bet_backtest_bets.csv", index=False)
    by_type.to_csv(OUTPUT_DIR / "multi_bet_backtest_summary.csv", index=False)
    (OUTPUT_DIR / "multi_bet_backtest_overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(overall, by_type, bets)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(by_type.sort_values("profit_yen", ascending=False).to_string(index=False))
    return overall, by_type, bets


def yen(value):
    return f"{int(value):,}円"


def pct(value):
    return "-" if value is None or pd.isna(value) else f"{float(value):.1%}"


def write_report(overall, by_type, bets):
    lines = [
        "# 券種別 損益バックテスト",
        "",
        f"- 作成日時: {overall['created_at_jst']}",
        f"- 検証レース数: {overall['history_races']:,}",
        f"- オッズ取得済みレース数: {overall['odds_races']:,}",
        f"- 全体購入: {overall['overall_bets']:,}点",
        f"- 全体的中: {overall['overall_hits']:,}点",
        f"- 全体購入額: {yen(overall['overall_stake_yen'])}",
        f"- 全体払戻: {yen(overall['overall_return_yen'])}",
        f"- 全体損益: {yen(overall['overall_profit_yen'])}",
        f"- 全体回収率: {pct(overall['overall_roi'])}",
        "",
        "## 券種別",
        "",
        "|券種|買い目|的中|的中率|購入|払戻|損益|回収率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_type.sort_values("profit_yen", ascending=False).iterrows():
        lines.append(
            f"|{row['bet_label']}|{int(row['bets'])}|{int(row['hits'])}|{pct(row['hit_rate'])}|"
            f"{yen(row['stake_yen'])}|{yen(row['return_yen'])}|{yen(row['profit_yen'])}|{pct(row['roi'])}|"
        )
    if len(bets):
        top = bets.sort_values("profit_yen", ascending=False).head(15)
        lines += [
            "",
            "## 利益上位",
            "",
            "|日付|場|R|券種|買い目|購入|オッズ|結果|損益|",
            "|---|---|---:|---|---|---:|---:|---|---:|",
        ]
        for _, row in top.iterrows():
            result = "的中" if row["is_hit"] else f"外れ({row.get('actual_buy', '')})"
            lines.append(
                f"|{row['date']}|{row['venue']}|{int(row['race_no'])}|{row['bet_label']}|{row['buy']}|"
                f"{int(row['stake_yen'])}|{float(row['odds_used']):.1f}|{result}|{int(row['profit_yen'])}|"
            )
    (OUTPUT_DIR / "multi_bet_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-dates", type=int, default=30)
    parser.add_argument("--retrain-every-days", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-prob", type=float, default=0.02)
    parser.add_argument("--min-expected-profit", type=float, default=100)
    parser.add_argument("--max-odds", type=float, default=300)
    parser.add_argument("--max-bets-per-race-type", type=int, default=2)
    parser.add_argument("--max-bets-per-day-type", type=int, default=40)
    parser.add_argument("--base-stake", type=int, default=100)
    parser.add_argument("--max-stake", type=int, default=500)
    args = parser.parse_args()
    run_multi_bet_backtest(
        args.min_train_dates,
        args.retrain_every_days,
        args.top_k,
        args.min_prob,
        args.min_expected_profit,
        args.max_odds,
        args.max_bets_per_race_type,
        args.max_bets_per_day_type,
        args.base_stake,
        args.max_stake,
    )


if __name__ == "__main__":
    main()
