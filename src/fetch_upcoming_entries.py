import argparse
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from common import RAW_DIR, ensure_dirs
from fetch_today_entries import RACE_SCHEDULE_CSV, parse_race_page, save_today_frames

UPCOMING_COUNT_FILE = RAW_DIR / "upcoming_races_count.txt"
UPCOMING_METADATA_FILE = RAW_DIR / "upcoming_entries_metadata.json"


def select_upcoming_races(schedule, now_epoch, min_minutes=5, max_minutes=40):
    frame = schedule.copy()
    frame["close_at"] = pd.to_numeric(frame.get("close_at"), errors="coerce")
    seconds_to_close = frame["close_at"] - float(now_epoch)
    return frame[
        seconds_to_close.ge(float(min_minutes) * 60)
        & seconds_to_close.le(float(max_minutes) * 60)
    ].copy()


def fetch_upcoming(min_minutes=5, max_minutes=40, sleep_sec=0.2):
    ensure_dirs()
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if not RACE_SCHEDULE_CSV.exists():
        UPCOMING_COUNT_FILE.write_text("0", encoding="ascii")
        print(f"schedule not found: {RACE_SCHEDULE_CSV}")
        return 0

    schedule = pd.read_csv(RACE_SCHEDULE_CSV, dtype={"race_id": str})
    schedule = schedule[schedule["date"].astype(str).eq(now.strftime("%Y-%m-%d"))]
    upcoming = select_upcoming_races(schedule, now.timestamp(), min_minutes, max_minutes)
    UPCOMING_COUNT_FILE.write_text(str(len(upcoming)), encoding="ascii")
    if upcoming.empty:
        print("no races in the near-close window")
        return 0

    all_entries = []
    all_odds = []
    failures = []
    for row in upcoming.to_dict("records"):
        url = row.get("source_url")
        try:
            entries, odds = parse_race_page(url)
            race_id = str(row["race_id"])
            all_entries.extend(item for item in entries if str(item.get("race_id")) == race_id)
            all_odds.extend(item for item in odds if str(item.get("race_id")) == race_id)
        except Exception as error:
            failures.append({"race_id": row.get("race_id"), "url": url, "error": str(error)})
        time.sleep(sleep_sec)

    if not all_entries:
        UPCOMING_COUNT_FILE.write_text("0", encoding="ascii")
        raise ValueError(f"failed to fetch upcoming entries: {failures[:3]}")

    entries, odds, _ = save_today_frames(all_entries, all_odds)
    fetched_races = int(entries["race_id"].nunique())
    UPCOMING_COUNT_FILE.write_text(str(fetched_races), encoding="ascii")
    metadata = {
        "fetched_at_jst": now.isoformat(timespec="seconds"),
        "min_minutes_to_close": min_minutes,
        "max_minutes_to_close": max_minutes,
        "scheduled_races": int(len(upcoming)),
        "fetched_races": fetched_races,
        "entry_rows": int(len(entries)),
        "odds_rows": int(len(odds)),
        "failures": failures,
    }
    UPCOMING_METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return fetched_races


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=float, default=5)
    parser.add_argument("--max-minutes", type=float, default=40)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()
    return fetch_upcoming(args.min_minutes, args.max_minutes, args.sleep_sec)


if __name__ == "__main__":
    main()
