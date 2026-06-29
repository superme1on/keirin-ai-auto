import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from common import ensure_dirs, RAW_DIR, TODAY_CSV

BASE_URL = "https://www.winticket.jp"
RACECARD_URL = f"{BASE_URL}/keirin/racecard"
TRIFECTA_ODDS_CSV = RAW_DIR / "today_trifecta_odds.csv"


def http_get(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; keirin-ai-auto/1.0)",
        "Accept-Language": "ja,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def extract_preloaded_state(html):
    m = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});\s*window\.__CONFIG__", html, re.S)
    if not m:
        raise ValueError("WINTICKET preloaded state was not found")
    return json.loads(m.group(1))


def tanstack_queries(state):
    return state.get("tanStackQuery", {}).get("queries", [])


def find_query_data(state, marker):
    for q in tanstack_queries(state):
        if marker in json.dumps(q.get("queryKey"), ensure_ascii=False):
            data = q.get("state", {}).get("data")
            if isinstance(data, dict):
                return data
    return {}


def collect_race_links(index_html, race_date):
    links = set()
    for href in re.findall(r'href="([^"]+)"', index_html):
        if re.fullmatch(r"/keirin/[^/]+/racecard/\d{10}/\d+/\d+", href):
            links.add(urljoin(BASE_URL, href))
    return sorted(links)


def normalize_date(yyyymmdd):
    s = str(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def recent_avg_finish(record):
    orders = []
    for key in ["currentCupResults", "previousCupResults"]:
        for result in record.get(key, []) or []:
            order = result.get("order")
            if isinstance(order, (int, float)) and order > 0:
                orders.append(order)
            if len(orders) >= 10:
                break
        if len(orders) >= 10:
            break
    return float(np.mean(orders)) if orders else np.nan


def days_since_last_race(record, race_date):
    dates = []
    for key in ["currentCupResults", "previousCupResults"]:
        for result in record.get(key, []) or []:
            m = re.search(r"(20\d{6})", str(result.get("raceId", "")))
            if m:
                dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    if not dates:
        return np.nan
    current = datetime.strptime(race_date, "%Y-%m-%d").date()
    before = [d for d in dates if d <= current]
    if not before:
        return np.nan
    return max((current - max(before)).days, 0)


def get_venue_name(state, race_data):
    cup_data = find_query_data(state, "FETCH_KEIRIN_CUP_RACES")
    venue = cup_data.get("venue")
    if isinstance(venue, dict) and venue.get("name"):
        return venue["name"]

    venue_id = None
    schedule = race_data.get("schedule", {})
    cup_id = schedule.get("cupId")
    for cup in race_data.get("cups", []):
        if cup.get("id") == cup_id:
            venue_id = cup.get("venueId")
            break
    venue_list = find_query_data(state, "FETCH_KEIRIN_VENUE_LIST")
    for venue in venue_list.get("venues", []):
        if str(venue.get("id")) == str(venue_id):
            return venue.get("name", "")
    for region in venue_list.get("regions", []):
        for venue in region.get("venues", []):
            if str(venue.get("id")) == str(venue_id):
                return venue.get("name", "")
    return ""


def parse_race_page(url):
    html = http_get(url)
    state = extract_preloaded_state(html)
    race_data = find_query_data(state, "FETCH_KEIRIN_RACE")
    if not race_data:
        raise ValueError(f"race data not found: {url}")

    schedule = race_data["schedule"]
    race = race_data["race"]
    race_date = normalize_date(schedule["date"])
    venue = get_venue_name(state, race_data)
    race_no = int(race["number"])
    race_id = str(race["id"])

    players = {str(p.get("id")): p for p in race_data.get("players", [])}
    records = {str(r.get("playerId")): r for r in race_data.get("records", [])}

    entry_rows = []
    for entry in race_data.get("entries", []):
        if entry.get("absent"):
            continue
        player_id = str(entry.get("playerId", ""))
        player = players.get(player_id, {})
        record = records.get(player_id, {})

        first_rate = pd.to_numeric(record.get("firstRate"), errors="coerce") / 100
        second_rate = pd.to_numeric(record.get("secondRate"), errors="coerce") / 100
        third_rate = pd.to_numeric(record.get("thirdRate"), errors="coerce") / 100

        entry_rows.append(
            {
                "race_id": race_id,
                "date": race_date,
                "venue": venue,
                "race_no": race_no,
                "player_id": player_id,
                "car_no": int(entry.get("number")),
                "age": player.get("age", np.nan),
                "score": record.get("racePoint", np.nan),
                "win_rate": first_rate,
                "place2_rate": second_rate,
                "place3_rate": third_rate,
                "back_count": record.get("back", np.nan),
                "style": record.get("style", ""),
                "recent_avg_finish": recent_avg_finish(record),
                "days_since_last_race": days_since_last_race(record, race_date),
                "venue_win_rate": first_rate,
                "odds_win": np.nan,
                "source_url": url,
                "player_name": player.get("name", ""),
                "race_class": race.get("class", ""),
                "race_type": race.get("raceType", ""),
                "distance": race.get("distance", np.nan),
            }
        )

    odds_data = find_query_data(state, "FETCH_KEIRIN_RACE_ODDS")
    odds_rows = []
    for item in odds_data.get("trifecta", []) or []:
        key = item.get("key", [])
        if len(key) != 3 or item.get("absent"):
            continue
        odds_rows.append(
            {
                "date": race_date,
                "venue": venue,
                "race_no": race_no,
                "race_id": race_id,
                "buy": f"{int(key[0])}-{int(key[1])}-{int(key[2])}",
                "trifecta_odds": item.get("odds", np.nan),
                "popularity_order": item.get("popularityOrder", np.nan),
                "source_url": url,
            }
        )

    return entry_rows, odds_rows


def fetch_today_entries(race_date=None, max_races=None, sleep_sec=0.2):
    ensure_dirs()
    race_date = race_date or datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    index_html = http_get(RACECARD_URL)
    links = collect_race_links(index_html, race_date)
    if max_races:
        links = links[:max_races]
    if not links:
        raise ValueError(f"no WINTICKET racecard links found for {race_date}")

    all_entries = []
    all_odds = []
    failures = []
    for i, link in enumerate(links, start=1):
        try:
            entries, odds = parse_race_page(link)
            entries = [r for r in entries if r.get("date") == race_date]
            odds = [r for r in odds if r.get("date") == race_date]
            if not entries:
                print(f"skipped {i}/{len(links)}: {link} date mismatch")
                continue
            all_entries.extend(entries)
            all_odds.extend(odds)
            print(f"fetched {i}/{len(links)}: {link} entries={len(entries)} odds={len(odds)}")
        except Exception as e:
            failures.append({"url": link, "error": str(e)})
            print(f"failed {i}/{len(links)}: {link} error={e}")
        time.sleep(sleep_sec)

    if not all_entries:
        raise ValueError(f"failed to fetch any entries for {race_date}: {failures[:3]}")

    entries_df = pd.DataFrame(all_entries).sort_values(["date", "venue", "race_no", "car_no"])
    entries_df.to_csv(TODAY_CSV, index=False)

    odds_df = pd.DataFrame(all_odds)
    if len(odds_df):
        odds_df = odds_df.sort_values(["date", "venue", "race_no", "popularity_order", "buy"])
    odds_df.to_csv(TRIFECTA_ODDS_CSV, index=False)

    metadata = {
        "source": "WINTICKET racecard",
        "source_url": RACECARD_URL,
        "date": race_date,
        "fetched_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "races": len(links),
        "entry_rows": int(len(entries_df)),
        "trifecta_odds_rows": int(len(odds_df)),
        "failures": failures,
    }
    (RAW_DIR / "today_entries_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return entries_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Race date in YYYY-MM-DD. Defaults to today in Asia/Tokyo.")
    parser.add_argument("--max-races", type=int, default=None)
    args = parser.parse_args()
    fetch_today_entries(args.date, args.max_races)


if __name__ == "__main__":
    main()
