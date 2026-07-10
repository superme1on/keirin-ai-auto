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

from common import ensure_dirs, RAW_DIR, TODAY_CSV, TODAY_ODDS_CSV
from race_features import build_entry_rows

BASE_URL = "https://www.winticket.jp"
RACECARD_URL = f"{BASE_URL}/keirin/racecard"
TRIFECTA_ODDS_CSV = RAW_DIR / "today_trifecta_odds.csv"
ODDS_COLUMNS = [
    "date",
    "venue",
    "race_no",
    "race_id",
    "bet_type",
    "buy",
    "odds",
    "min_odds",
    "max_odds",
    "odds_used",
    "popularity_order",
    "source_url",
]
TRIFECTA_ODDS_COLUMNS = ["date", "venue", "race_no", "race_id", "buy", "trifecta_odds", "popularity_order", "source_url"]
BET_SPECS = {
    "trifecta": {"source": "trifecta", "key_len": 3, "ordered": True},
    "trio": {"source": "trio", "key_len": 3, "ordered": False},
    "exacta": {"source": "exacta", "key_len": 2, "ordered": True},
    "quinella": {"source": "quinella", "key_len": 2, "ordered": False},
    "quinella_place": {"source": "quinellaPlace", "key_len": 2, "ordered": False},
    "bracket_exacta": {"source": "bracketExacta", "key_len": 2, "ordered": True},
    "bracket_quinella": {"source": "bracketQuinella", "key_len": 2, "ordered": False},
}


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


def key_to_buy(key, ordered=True):
    values = [int(x) for x in key]
    if not ordered:
        values = sorted(values)
    return "-".join(str(x) for x in values)


def build_odds_rows(odds_data, race_date, venue, race_no, race_id, url):
    rows = []
    for bet_type, spec in BET_SPECS.items():
        for item in odds_data.get(spec["source"], []) or []:
            key = item.get("key", [])
            if len(key) != spec["key_len"] or item.get("absent"):
                continue
            odds = pd.to_numeric(item.get("odds"), errors="coerce")
            min_odds = pd.to_numeric(item.get("minOdds"), errors="coerce")
            max_odds = pd.to_numeric(item.get("maxOdds"), errors="coerce")
            odds_used = odds
            if bet_type == "quinella_place" and (pd.isna(odds_used) or odds_used <= 0):
                odds_used = min_odds
            rows.append(
                {
                    "date": race_date,
                    "venue": venue,
                    "race_no": race_no,
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "buy": key_to_buy(key, ordered=spec.get("ordered", True)),
                    "odds": odds,
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                    "odds_used": odds_used,
                    "popularity_order": item.get("popularityOrder", np.nan),
                    "source_url": url,
                }
            )
    return rows


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

    entry_rows = build_entry_rows(
        race_data,
        race_date=race_date,
        venue=venue,
        race_no=race_no,
        race_id=race_id,
        source_url=url,
        include_results=False,
    )

    odds_data = find_query_data(state, "FETCH_KEIRIN_RACE_ODDS")
    odds_rows = build_odds_rows(odds_data, race_date, venue, race_no, race_id, url)

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

    odds_df = pd.DataFrame(all_odds, columns=ODDS_COLUMNS)
    if len(odds_df):
        odds_df = odds_df.sort_values(["date", "venue", "race_no", "bet_type", "popularity_order", "buy"])
    odds_df.to_csv(TODAY_ODDS_CSV, index=False)

    trifecta_df = odds_df[odds_df["bet_type"].eq("trifecta")].copy() if len(odds_df) else pd.DataFrame(columns=ODDS_COLUMNS)
    if len(trifecta_df):
        trifecta_df["trifecta_odds"] = trifecta_df["odds_used"]
        trifecta_df = trifecta_df[TRIFECTA_ODDS_COLUMNS]
    else:
        trifecta_df = pd.DataFrame(columns=TRIFECTA_ODDS_COLUMNS)
    trifecta_df.to_csv(TRIFECTA_ODDS_CSV, index=False)

    metadata = {
        "source": "WINTICKET racecard",
        "source_url": RACECARD_URL,
        "date": race_date,
        "fetched_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "races": len(links),
        "entry_rows": int(len(entries_df)),
        "odds_rows": int(len(odds_df)),
        "bet_types": sorted(odds_df["bet_type"].dropna().unique().tolist()) if len(odds_df) else [],
        "trifecta_odds_rows": int(len(trifecta_df)),
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
