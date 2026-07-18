import argparse
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, ensure_dirs

GROWTH_LOG_CSV = OUTPUT_DIR / "growth_log.csv"
TRIAL_REPORT_MD = OUTPUT_DIR / "trial_report.md"
SHADOW_OVERALL_JSON = OUTPUT_DIR / "shadow_overall.json"


def fmt_yen(value):
    if pd.isna(value):
        return "-"
    return f"{int(value):,}円"


def fmt_pct(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def parse_logged_at(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize("Asia/Tokyo")
    return parsed.dt.tz_convert("Asia/Tokyo")


def build_trial_report(days=31, start_date=None):
    ensure_dirs()
    if not GROWTH_LOG_CSV.exists():
        raise FileNotFoundError(f"missing {GROWTH_LOG_CSV}")

    df = pd.read_csv(GROWTH_LOG_CSV)
    if df.empty:
        raise ValueError("growth_log.csv is empty")

    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    cutoff = now - timedelta(days=days)
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("Asia/Tokyo"))
        cutoff = max(cutoff, start)
    df["logged_at_dt"] = parse_logged_at(df["logged_at_jst"])

    settlements = df[(df["kind"].eq("settle")) & (df["logged_at_dt"] >= cutoff)].copy()
    if start_date and "target_date" in settlements.columns:
        target_dates = pd.to_datetime(settlements["target_date"], errors="coerce").dt.date
        settlements = settlements[target_dates >= datetime.strptime(start_date, "%Y-%m-%d").date()].copy()
    settlements = settlements.sort_values("logged_at_dt").drop_duplicates(["target_date"], keep="last")
    training = df[(df["kind"].eq("train")) & (df["logged_at_dt"] >= cutoff)].copy()
    backtests = df[(df["kind"].eq("multi_bet_backtest")) & (df["logged_at_dt"] >= cutoff)].copy()

    stake = pd.to_numeric(settlements.get("stake_yen", 0), errors="coerce").fillna(0).sum()
    returns = pd.to_numeric(settlements.get("return_yen", 0), errors="coerce").fillna(0).sum()
    profit = pd.to_numeric(settlements.get("profit_yen", 0), errors="coerce").fillna(0).sum()
    bets = pd.to_numeric(settlements.get("settlement_bets", 0), errors="coerce").fillna(0).sum()
    hits = pd.to_numeric(settlements.get("settlement_hits", 0), errors="coerce").fillna(0).sum()
    roi = profit / stake if stake else np.nan
    return_rate = returns / stake if stake else np.nan
    hit_rate = hits / bets if bets else np.nan

    lines = [
        "# 競輪AI 1か月トライアル記録",
        "",
        f"作成日時: {now.isoformat(timespec='seconds')}",
        f"対象期間: {cutoff.date()} から直近{days}日",
        "",
        "## 実購入ベース",
        "",
        f"- 決済回数: {len(settlements):,}回",
        f"- 購入点数: {int(bets):,}点",
        f"- 的中点数: {int(hits):,}点",
        f"- 的中率: {fmt_pct(hit_rate)}",
        f"- 購入金額: {fmt_yen(stake)}",
        f"- 払戻金額: {fmt_yen(returns)}",
        f"- 損益: {fmt_yen(profit)}",
        f"- 回収率: {fmt_pct(return_rate)}",
        f"- ROI: {fmt_pct(roi)}",
        "",
    ]

    if SHADOW_OVERALL_JSON.exists():
        shadow = json.loads(SHADOW_OVERALL_JSON.read_text(encoding="utf-8"))
        lines += [
            "## 影予想（購入せず検証）",
            "",
            f"- 保存候補: {int(shadow.get('bets_recorded', shadow.get('bets_total', 0))):,}点",
            f"- 締切前の有効予想: {int(shadow.get('bets_total', 0)):,}点",
            f"- 締切後除外: {int(shadow.get('bets_excluded_after_close', 0)):,}点",
            f"- 確定点数: {int(shadow.get('bets_decided', 0)):,}点",
            f"- 的中点数: {int(shadow.get('hits', 0)):,}点",
            f"- 仮想購入金額: {fmt_yen(shadow.get('stake_yen', np.nan))}",
            f"- 公式払戻金額: {fmt_yen(shadow.get('return_yen', np.nan))}",
            f"- 損益: {fmt_yen(shadow.get('profit_yen', np.nan))}",
            f"- 回収率: {fmt_pct(shadow.get('return_rate', np.nan))}",
            f"- ROI: {fmt_pct(shadow.get('roi', np.nan))}",
            "",
        ]

    lines += [
        "## 最新ログ",
        "",
    ]

    display_df = df[df["logged_at_dt"] >= cutoff].copy()
    if start_date and "target_date" in display_df.columns:
        target_dates = pd.to_datetime(display_df["target_date"], errors="coerce").dt.date
        start_day = datetime.strptime(start_date, "%Y-%m-%d").date()
        keep = display_df["kind"].ne("settle") | target_dates.isna() | (target_dates >= start_day)
        display_df = display_df[keep].copy()
    latest = display_df.sort_values("logged_at_dt").tail(10).copy()
    lines += [
        "| 日時 | 種別 | 対象日 | 購入 | 的中 | 投資 | 払戻 | 損益 | ROI | メモ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in latest.iterrows():
        lines.append(
            "| "
            f"{row.get('logged_at_jst', '')} | {row.get('kind', '')} | {row.get('target_date', '')} | "
            f"{'' if pd.isna(row.get('settlement_bets', np.nan)) else int(row.get('settlement_bets', 0))} | "
            f"{'' if pd.isna(row.get('settlement_hits', np.nan)) else int(row.get('settlement_hits', 0))} | "
            f"{fmt_yen(row.get('stake_yen', np.nan))} | {fmt_yen(row.get('return_yen', np.nan))} | "
            f"{fmt_yen(row.get('profit_yen', np.nan))} | {fmt_pct(row.get('roi', np.nan))} | {row.get('note', '')} |"
        )

    lines += [
        "",
        "## 学習状況",
        "",
        f"- 期間内の再学習回数: {len(training):,}回",
        f"- 期間内の券種別バックテスト回数: {len(backtests):,}回",
    ]
    if len(training):
        last_train = training.sort_values("logged_at_dt").iloc[-1]
        lines.append(f"- 最新AUC: {last_train.get('auc_binary', np.nan):.4f}")
        lines.append(f"- 最新本命的中率: {fmt_pct(last_train.get('top1_hit_rate', np.nan))}")

    TRIAL_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved: {TRIAL_REPORT_MD}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--start-date", help="YYYY-MM-DD. Ignore logs before this date.")
    args = parser.parse_args()
    build_trial_report(args.days, args.start_date)


if __name__ == "__main__":
    main()
