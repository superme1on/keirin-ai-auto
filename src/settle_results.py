import argparse
import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, RAW_DIR, ensure_dirs
from fetch_today_entries import extract_preloaded_state, find_query_data, http_get

TODAY_CSV = RAW_DIR / "today_entries.csv"
LATEST_BETS_CSV = OUTPUT_DIR / "latest_bets.csv"
SETTLED_BETS_CSV = OUTPUT_DIR / "settled_bets.csv"
PURCHASE_PLAN_CSV = OUTPUT_DIR / "purchase_plan.csv"
SETTLEMENT_SUMMARY_CSV = OUTPUT_DIR / "settlement_summary.csv"
REPORT_MD = OUTPUT_DIR / "japanese_report.md"


def key_to_buy(values, ordered=True):
    values = [int(x) for x in values]
    if not ordered:
        values = sorted(values)
    return "-".join(str(x) for x in values)


def actual_buy_sets(top3, bracket_by_car):
    cars = [int(x) for x in top3[:3]]
    top2 = cars[:2]
    brackets = [bracket_by_car.get(car) for car in cars]
    bracket_top2 = [b for b in brackets[:2] if b is not None]
    actual = {
        "trifecta": key_to_buy(cars, ordered=True) if len(cars) == 3 else "",
        "trio": key_to_buy(cars, ordered=False) if len(cars) == 3 else "",
        "exacta": key_to_buy(top2, ordered=True) if len(top2) == 2 else "",
        "quinella": key_to_buy(top2, ordered=False) if len(top2) == 2 else "",
        "bracket_exacta": key_to_buy(bracket_top2, ordered=True) if len(bracket_top2) == 2 else "",
        "bracket_quinella": key_to_buy(bracket_top2, ordered=False) if len(bracket_top2) == 2 else "",
    }
    if len(cars) == 3:
        actual["quinella_place"] = "|".join(
            sorted(
                [
                    key_to_buy([cars[0], cars[1]], ordered=False),
                    key_to_buy([cars[0], cars[2]], ordered=False),
                    key_to_buy([cars[1], cars[2]], ordered=False),
                ]
            )
        )
    else:
        actual["quinella_place"] = ""
    return actual


def parse_result_page(url):
    html = http_get(url)
    state = extract_preloaded_state(html)
    race_data = find_query_data(state, "FETCH_KEIRIN_RACE")
    if not race_data:
        raise ValueError(f"race data not found: {url}")

    race = race_data["race"]
    race_id = str(race["id"])
    entry_by_player = {str(e.get("playerId")): int(e.get("number")) for e in race_data.get("entries", [])}
    bracket_by_car = {
        int(e.get("number")): int(e.get("bracketNumber"))
        for e in race_data.get("entries", [])
        if e.get("number") and e.get("bracketNumber")
    }

    ordered = []
    for result in race_data.get("results", []) or []:
        order = result.get("order")
        player_id = str(result.get("playerId"))
        car_no = entry_by_player.get(player_id)
        if isinstance(order, int) and car_no:
            ordered.append((order, car_no, player_id))
    ordered = sorted(ordered)
    top3 = [car_no for order, car_no, player_id in ordered if order in [1, 2, 3]][:3]
    winning_buy = "-".join(str(x) for x in top3) if len(top3) == 3 else ""
    actual = actual_buy_sets(top3, bracket_by_car)

    winning_odds = np.nan
    for item in race_data.get("trifecta", []) or []:
        key = item.get("key", [])
        buy = "-".join(str(int(x)) for x in key) if len(key) == 3 else ""
        if buy == winning_buy:
            winning_odds = pd.to_numeric(item.get("odds"), errors="coerce")
            break

    result = {
        "race_id": race_id,
        "actual_trifecta": winning_buy,
        "actual_trifecta_odds": winning_odds,
        "race_status": race.get("status"),
        "decided_at": race.get("decidedAt"),
        "source_url": url,
    }
    for bet_type, buy in actual.items():
        result[f"actual_{bet_type}"] = buy
    return result


def fetch_results(sleep_sec=0.2):
    entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    if "source_url" not in entries.columns:
        raise ValueError("today_entries.csv does not have source_url column")

    rows = []
    for i, url in enumerate(entries["source_url"].dropna().drop_duplicates(), start=1):
        try:
            result = parse_result_page(url)
            rows.append(result)
            print(f"settled source {i}: {url} actual={result['actual_trifecta']}")
        except Exception as e:
            print(f"failed source {i}: {url} error={e}")
        time.sleep(sleep_sec)

    if not rows:
        raise ValueError("no race results were fetched")
    return pd.DataFrame(rows)


def summarize(name, bets):
    if len(bets) == 0:
        return {
            "target": name,
            "bets": 0,
            "hits": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": np.nan,
        }

    stake = pd.to_numeric(bets["stake_yen"], errors="coerce").fillna(0).sum()
    returns = pd.to_numeric(bets["actual_return_yen"], errors="coerce").fillna(0).sum()
    profit = pd.to_numeric(bets["actual_profit_yen"], errors="coerce").fillna(0).sum()
    return {
        "target": name,
        "bets": int(len(bets)),
        "hits": int(bets["is_hit"].sum()),
        "stake_yen": int(stake),
        "return_yen": int(returns),
        "profit_yen": int(profit),
        "roi": float(profit / stake) if stake else np.nan,
    }


def build_report(selected, summary):
    lines = [
        "# AI予想 結果レポート",
        "",
        f"作成日時: {datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')}",
        "",
        "## 買う条件",
        "",
        "- 3連単候補のうち `expected_profit_yen > 0` のものだけを購入",
        "- 1点あたり100円",
        "- 理由: AIの3連単近似確率とWINTICKET実オッズを掛けて、期待利益がプラスだったため",
        "",
        "## 損益まとめ",
        "",
    ]
    for row in summary.to_dict("records"):
        roi = "" if pd.isna(row["roi"]) else f"{row['roi']:.2%}"
        lines.append(
            f"- {row['target']}: {row['bets']}点購入 / 的中 {row['hits']}点 / "
            f"投資 {row['stake_yen']:,}円 / 払戻 {row['return_yen']:,}円 / "
            f"損益 {row['profit_yen']:,}円 / 回収率 {roi}"
        )

    lines += [
        "",
        "## 購入候補 上位20点",
        "",
        "| 日付 | 場 | R | 買い目 | 確率 | オッズ | 期待利益 | 結果 | 損益 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    top = selected.sort_values(["date", "venue", "race_no", "candidate_rank"]).head(20)
    for _, row in top.iterrows():
        result = "的中" if row["is_hit"] else f"外れ({row.get('actual_for_bet_type', row.get('actual_trifecta', ''))})"
        prob = row.get("prob", row.get("trifecta_prob_approx", np.nan))
        odds = row.get("odds_used", row.get("trifecta_odds", np.nan))
        lines.append(
            f"| {row['date']} | {row['venue']} | {row['race_no']} | {row['buy']} | "
            f"{prob:.3f} | {odds:.1f} | "
            f"{int(row['expected_profit_yen']):,}円 | {result} | {int(row['actual_profit_yen']):,}円 |"
        )

    lines += [
        "",
        "詳しい全件は `outputs/purchase_plan.csv` と `outputs/settled_bets.csv` を見てください。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-expected-profit", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dirs()
    results = fetch_results()
    bets = pd.read_csv(LATEST_BETS_CSV, dtype={"race_id": str})
    if "bet_type" not in bets.columns:
        bets["bet_type"] = "trifecta"

    settled = bets.merge(results, on="race_id", how="left")
    settled["is_selected"] = pd.to_numeric(settled["expected_profit_yen"], errors="coerce").fillna(-10**9) > args.min_expected_profit
    settled["actual_for_bet_type"] = settled.apply(
        lambda row: row.get(f"actual_{row.get('bet_type', 'trifecta')}", row.get("actual_trifecta", "")),
        axis=1,
    )
    settled["is_hit"] = settled.apply(
        lambda row: str(row["buy"]) in str(row.get("actual_for_bet_type", "")).split("|"),
        axis=1,
    )
    return_col = "return_if_hit_yen" if "return_if_hit_yen" in settled.columns else "trifecta_return_yen"
    settled["actual_return_yen"] = np.where(
        settled["is_hit"],
        pd.to_numeric(settled.get(return_col, 0), errors="coerce").fillna(0),
        0,
    )
    settled["actual_profit_yen"] = settled["actual_return_yen"] - pd.to_numeric(settled["stake_yen"], errors="coerce").fillna(0)
    settled.to_csv(SETTLED_BETS_CSV, index=False)

    selected = settled[settled["is_selected"]].copy()
    selected.to_csv(PURCHASE_PLAN_CSV, index=False)

    summary = pd.DataFrame(
        [
            summarize("AI購入分 expected_profit_yen > 0", selected),
            summarize("参考: latest_bets全候補", settled),
        ]
    )
    summary.to_csv(SETTLEMENT_SUMMARY_CSV, index=False)
    REPORT_MD.write_text(build_report(selected, summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"saved: {PURCHASE_PLAN_CSV}")
    print(f"saved: {SETTLED_BETS_CSV}")
    print(f"saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
