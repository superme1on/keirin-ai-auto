import argparse
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_history import extract_month_cups
from common import MODEL_PATH, OUTPUT_DIR, ensure_dirs
from fetch_today_entries import extract_preloaded_state, find_query_data, http_get
from settle_results import parse_result_page, settled_return_yen

SHADOW_GLOB = "shadow_bets_20*.csv"
RESULT_CACHE_CSV = OUTPUT_DIR / "shadow_race_results.csv"
SETTLED_CSV = OUTPUT_DIR / "shadow_settled_bets.csv"
DAILY_SUMMARY_CSV = OUTPUT_DIR / "shadow_daily_summary.csv"
BET_TYPE_SUMMARY_CSV = OUTPUT_DIR / "shadow_bet_type_summary.csv"
OVERALL_JSON = OUTPUT_DIR / "shadow_overall.json"
REPORT_MD = OUTPUT_DIR / "shadow_report.md"
TARGET_ROI = 0.50
MIN_BETS_FOR_TARGET_JUDGMENT = 500
MIN_SECONDS_BEFORE_CLOSE = 60


def parse_race_id(race_id):
    match = re.fullmatch(r"(?P<race_no>\d{2})(?P<venue_id>\d{2})(?P<date>20\d{6})", str(race_id))
    if not match:
        raise ValueError(f"unsupported race_id: {race_id}")
    return {
        "race_id": str(race_id),
        "race_no": int(match.group("race_no")),
        "venue_id": match.group("venue_id"),
        "date": match.group("date"),
    }


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_file_created_at(path):
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%aI", "--", str(path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    timestamps = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not timestamps:
        return ""
    parsed = pd.to_datetime(timestamps[-1], errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.tz_convert("Asia/Tokyo").isoformat()


def load_shadow_bets(start_date=None, end_date=None):
    frames = []
    for path in sorted(OUTPUT_DIR.glob(SHADOW_GLOB)):
        frame = pd.read_csv(path, dtype={"race_id": str, "buy": str, "bet_type": str})
        if frame.empty:
            continue
        frame["prediction_file"] = path.name
        created_at = git_file_created_at(path)
        if "prediction_created_at_jst" not in frame.columns:
            frame["prediction_created_at_jst"] = created_at
        else:
            frame["prediction_created_at_jst"] = frame["prediction_created_at_jst"].fillna("").replace("", created_at)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    bets = pd.concat(frames, ignore_index=True, sort=False)
    bets["date"] = bets["date"].astype(str)
    if start_date:
        bets = bets[bets["date"].ge(start_date)]
    if end_date:
        bets = bets[bets["date"].le(end_date)]
    key = ["date", "race_id", "bet_type", "buy"]
    return bets.drop_duplicates(key, keep="last").sort_values(["date", "venue", "race_no", "bet_type", "buy"])


def relevant_schedule_months(race_metadata):
    months = set()
    for item in race_metadata.values():
        event_date = datetime.strptime(item["date"], "%Y%m%d").date()
        months.add(event_date.strftime("%Y%m"))
        months.add((event_date.replace(day=1) - timedelta(days=1)).strftime("%Y%m"))
    return sorted(months)


def retry_call(function, argument, attempts=4):
    last_error = None
    for attempt in range(attempts):
        try:
            return function(argument)
        except Exception as error:
            last_error = error
            if attempt + 1 >= attempts:
                break
            wait_seconds = 1.0 * (2**attempt)
            if "429" not in str(error):
                wait_seconds = min(wait_seconds, 2.0)
            time.sleep(wait_seconds)
    raise last_error


def collect_cup_race_url_map(cup):
    html = http_get(cup["cup_url"])
    state = extract_preloaded_state(html)
    data = find_query_data(state, "FETCH_KEIRIN_CUP")
    if not data:
        data = find_query_data(state, "FETCH_KEIRIN_CUP_RACES")
    schedules = {str(item.get("id")): item for item in data.get("schedules", [])}
    url_map = {}
    for race in data.get("races", []):
        schedule = schedules.get(str(race.get("scheduleId")), {})
        race_id = str(race.get("id", ""))
        index = schedule.get("index")
        race_no = race.get("number")
        if race_id and index and race_no:
            url_map[race_id] = f"{cup['cup_url'].rstrip('/')}/{index}/{race_no}"
    return url_map


def resolve_race_urls(race_ids, workers=4):
    metadata = {str(race_id): parse_race_id(race_id) for race_id in race_ids}
    cups = []
    for year_month in relevant_schedule_months(metadata):
        cups.extend(extract_month_cups(year_month))

    relevant_cups = {}
    for cup in cups:
        cup_url = cup.get("cup_url")
        venue_id = str(cup.get("venueId", "")).zfill(2)
        start_date = str(cup.get("startDate", ""))
        end_date = str(cup.get("endDate", ""))
        if not cup_url:
            continue
        if any(
            item["venue_id"] == venue_id and start_date <= item["date"] <= end_date
            for item in metadata.values()
        ):
            relevant_cups[cup_url] = cup

    url_map = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {
            executor.submit(retry_call, collect_cup_race_url_map, cup): cup
            for cup in relevant_cups.values()
        }
        for future in as_completed(futures):
            cup = futures[future]
            try:
                url_map.update(future.result())
            except Exception as error:
                failures.append({"url": cup.get("cup_url", ""), "error": str(error)})

    missing = sorted(set(metadata) - set(url_map))
    return {race_id: url_map[race_id] for race_id in metadata if race_id in url_map}, missing, failures


def load_result_cache():
    if not RESULT_CACHE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULT_CACHE_CSV, dtype={"race_id": str})


def fetch_results(race_urls, workers=2):
    cache = load_result_cache()
    completed_ids = set()
    if not cache.empty and {"actual_trifecta", "close_at"}.issubset(cache.columns):
        has_close = pd.to_numeric(cache["close_at"], errors="coerce").fillna(0).gt(0)
        completed_ids = set(
            cache.loc[
                cache["actual_trifecta"].fillna("").astype(str).str.len().gt(0) & has_close,
                "race_id",
            ].astype(str)
        )

    targets = {race_id: url for race_id, url in race_urls.items() if race_id not in completed_ids}
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {
            executor.submit(retry_call, parse_result_page, url): (race_id, url)
            for race_id, url in targets.items()
        }
        for future in as_completed(futures):
            race_id, url = futures[future]
            try:
                result = future.result()
                if str(result.get("race_id")) != str(race_id):
                    raise ValueError(f"race_id mismatch: expected={race_id} actual={result.get('race_id')}")
                rows.append(result)
            except Exception as error:
                failures.append({"race_id": race_id, "url": url, "error": str(error)})

    fetched = pd.DataFrame(rows)
    parts = [frame for frame in [cache, fetched] if not frame.empty]
    results = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if not results.empty:
        results = results.drop_duplicates("race_id", keep="last").sort_values("race_id")
        results.to_csv(RESULT_CACHE_CSV, index=False)
    return results, failures


def settle_shadow_rows(bets, results):
    if bets.empty:
        return bets.copy()
    result_frame = results.copy()
    if not result_frame.empty and "source_url" in result_frame.columns:
        result_frame = result_frame.rename(columns={"source_url": "result_source_url"})
    settled = bets.merge(result_frame, on="race_id", how="left")
    if "bet_type" not in settled.columns:
        settled["bet_type"] = "trifecta"
    settled["actual_for_bet_type"] = settled.apply(
        lambda row: row.get(f"actual_{row.get('bet_type', 'trifecta')}", row.get("actual_trifecta", "")),
        axis=1,
    )
    settled["is_decided"] = settled["actual_for_bet_type"].fillna("").astype(str).str.len().gt(0)
    settled["is_hit"] = settled.apply(
        lambda row: bool(row["is_decided"]) and str(row["buy"]) in str(row["actual_for_bet_type"]).split("|"),
        axis=1,
    )
    prediction_at = pd.to_datetime(settled.get("prediction_created_at_jst", ""), errors="coerce", utc=True)
    close_at = pd.to_datetime(
        pd.to_numeric(settled.get("close_at", np.nan), errors="coerce"),
        unit="s",
        errors="coerce",
        utc=True,
    )
    seconds_before_close = (close_at - prediction_at).dt.total_seconds()
    settled["seconds_before_close"] = seconds_before_close
    settled["is_prospective"] = seconds_before_close.ge(MIN_SECONDS_BEFORE_CLOSE)
    settled["evaluation_status"] = np.select(
        [
            settled["is_prospective"],
            prediction_at.notna() & close_at.notna(),
        ],
        [
            "締切前予想",
            "締切後のため除外",
        ],
        default="時刻不明のため除外",
    )
    fallback = pd.to_numeric(settled.get("return_if_hit_yen", 0), errors="coerce").fillna(0)
    stake = pd.to_numeric(settled.get("stake_yen", 0), errors="coerce").fillna(0)
    settled["actual_return_yen"] = [
        settled_return_yen(row, fallback_return)
        for (_, row), fallback_return in zip(settled.iterrows(), fallback.to_numpy())
    ]
    settled["actual_profit_yen"] = np.where(
        settled["is_decided"],
        settled["actual_return_yen"] - stake,
        np.nan,
    )
    return settled


def summarize_rows(frame):
    prospective = frame[frame["is_prospective"]].copy()
    decided = prospective[prospective["is_decided"]].copy()
    stake = pd.to_numeric(decided.get("stake_yen", 0), errors="coerce").fillna(0).sum()
    returns = pd.to_numeric(decided.get("actual_return_yen", 0), errors="coerce").fillna(0).sum()
    profit = returns - stake
    return {
        "bets_recorded": int(len(frame)),
        "bets_total": int(len(prospective)),
        "bets_decided": int(len(decided)),
        "bets_pending": int((~prospective["is_decided"]).sum()),
        "bets_excluded_after_close": int(
            frame["evaluation_status"].eq("締切後のため除外").sum()
        ),
        "bets_excluded_unknown_time": int(
            frame["evaluation_status"].eq("時刻不明のため除外").sum()
        ),
        "hits": int(decided["is_hit"].sum()),
        "hit_rate": float(decided["is_hit"].mean()) if len(decided) else None,
        "stake_yen": int(stake),
        "return_yen": int(returns),
        "profit_yen": int(profit),
        "return_rate": float(returns / stake) if stake else None,
        "roi": float(profit / stake) if stake else None,
    }


def build_summaries(settled):
    daily_rows = []
    for date, group in settled.groupby("date", sort=True):
        daily_rows.append({"date": date, **summarize_rows(group)})
    daily = pd.DataFrame(daily_rows)

    bet_type_rows = []
    for (bet_type, bet_label), group in settled.groupby(["bet_type", "bet_label"], sort=True, dropna=False):
        bet_type_rows.append(
            {
                "bet_type": bet_type,
                "bet_label": bet_label,
                **summarize_rows(group),
            }
        )
    bet_types = pd.DataFrame(bet_type_rows)
    return daily, bet_types


def fmt_pct(value):
    return "-" if value is None or pd.isna(value) else f"{float(value):.2%}"


def build_report(overall, daily, bet_types, settled):
    generated = overall["generated_at_jst"]
    lines = [
        "# 競輪AI 影予想 成長記録",
        "",
        f"更新日時: {generated}",
        "",
        "## 現在の判定",
        "",
        "- 実購入: 停止中",
        "- 評価方法: 購入せずに記録した予想を公式確定払戻で精算",
        f"- 目標ROI +50%の判定: {'合格' if overall['target_passed'] else '未達'}",
        f"- 判定に必要な最低確定点数: {MIN_BETS_FOR_TARGET_JUDGMENT:,}点",
        "",
        "## 累計",
        "",
        f"- 保存された候補: {overall['bets_recorded']:,}点",
        f"- 締切前の有効予想: {overall['bets_total']:,}点",
        f"- 締切後のため除外: {overall['bets_excluded_after_close']:,}点",
        f"- 時刻不明のため除外: {overall['bets_excluded_unknown_time']:,}点",
        f"- 確定: {overall['bets_decided']:,}点 / 未確定: {overall['bets_pending']:,}点",
        f"- 的中: {overall['hits']:,}点（{fmt_pct(overall['hit_rate'])}）",
        f"- 仮想購入額: {overall['stake_yen']:,}円",
        f"- 公式払戻額: {overall['return_yen']:,}円",
        f"- 損益: {overall['profit_yen']:,}円",
        f"- 回収率: {fmt_pct(overall['return_rate'])}",
        f"- ROI（純利益÷購入額）: {fmt_pct(overall['roi'])}",
        "",
        "## 日別",
        "",
        "| 日付 | 有効/保存 | 除外 | 確定 | 的中 | 購入 | 払戻 | 損益 | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in daily.sort_values("date").iterrows():
        lines.append(
            f"| {row['date']} | {int(row['bets_total'])}/{int(row['bets_recorded'])} | "
            f"{int(row['bets_excluded_after_close'] + row['bets_excluded_unknown_time'])} | "
            f"{int(row['bets_decided'])} | {int(row['hits'])} | {int(row['stake_yen']):,}円 | "
            f"{int(row['return_yen']):,}円 | {int(row['profit_yen']):,}円 | {fmt_pct(row['roi'])} |"
        )

    lines += [
        "",
        "## 券種別",
        "",
        "| 券種 | 確定 | 的中 | 購入 | 払戻 | 損益 | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bet_types.sort_values("bet_label").iterrows():
        lines.append(
            f"| {row['bet_label']} | {int(row['bets_decided'])} | {int(row['hits'])} | "
            f"{int(row['stake_yen']):,}円 | {int(row['return_yen']):,}円 | "
            f"{int(row['profit_yen']):,}円 | {fmt_pct(row['roi'])} |"
        )

    hits = settled[
        settled["is_prospective"] & settled["is_decided"] & settled["is_hit"]
    ].sort_values(["date", "venue", "race_no"])
    lines += [
        "",
        "## 的中した買い目",
        "",
        "| 日付 | 場 | R | 券種 | 買い目 | 購入 | 払戻 | 損益 |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    if hits.empty:
        lines.append("| - | - | - | - | - | - | - | - |")
    else:
        for _, row in hits.iterrows():
            lines.append(
                f"| {row['date']} | {row['venue']} | {int(row['race_no'])} | {row['bet_label']} | "
                f"{row['buy']} | {int(row['stake_yen']):,}円 | {int(row['actual_return_yen']):,}円 | "
                f"{int(row['actual_profit_yen']):,}円 |"
            )

    lines += [
        "",
        "朝取得したオッズでの影予想です。締切直前オッズとのずれがあるため、この記録だけで実購入は許可しません。",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    ensure_dirs()
    bets = load_shadow_bets(args.start_date, args.end_date)
    if bets.empty:
        raise ValueError("no shadow prediction rows were found")

    race_urls, missing_ids, resolve_failures = resolve_race_urls(bets["race_id"].unique(), args.workers)
    results, result_failures = fetch_results(race_urls, args.workers)
    settled = settle_shadow_rows(bets, results)
    settled.to_csv(SETTLED_CSV, index=False)

    daily, bet_types = build_summaries(settled)
    daily.to_csv(DAILY_SUMMARY_CSV, index=False)
    bet_types.to_csv(BET_TYPE_SUMMARY_CSV, index=False)

    summary = summarize_rows(settled)
    generated = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    overall = {
        "generated_at_jst": generated,
        **summary,
        "positive_days": int((pd.to_numeric(daily["profit_yen"], errors="coerce") > 0).sum()),
        "days_decided": int((pd.to_numeric(daily["bets_decided"], errors="coerce") > 0).sum()),
        "target_roi": TARGET_ROI,
        "minimum_bets_for_target_judgment": MIN_BETS_FOR_TARGET_JUDGMENT,
        "target_passed": bool(summary["bets_decided"] >= MIN_BETS_FOR_TARGET_JUDGMENT and summary["roi"] >= TARGET_ROI),
        "current_model_sha256": file_sha256(MODEL_PATH),
        "missing_race_ids": missing_ids,
        "failures": resolve_failures + result_failures,
    }
    OVERALL_JSON.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(build_report(overall, daily, bet_types, settled), encoding="utf-8")

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"saved: {SETTLED_CSV}")
    print(f"saved: {DAILY_SUMMARY_CSV}")
    print(f"saved: {REPORT_MD}")
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=2)
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
