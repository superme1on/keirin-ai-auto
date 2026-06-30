import argparse
import json
import re
import time
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, RAW_DIR, ensure_dirs
from fetch_today_entries import (
    BASE_URL,
    days_since_last_race,
    extract_preloaded_state,
    find_query_data,
    get_venue_name,
    http_get,
    normalize_date,
    recent_avg_finish,
)


def month_start(dt):
    return dt.strftime("%Y%m01")


def add_months(dt, months):
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    return dt.replace(year=y, month=m, day=1)


def extract_month_cups(year_month):
    url = f"{BASE_URL}/keirin/schedules/{year_month}"
    html = http_get(url)
    state = extract_preloaded_state(html)
    data = find_query_data(state, "FETCH_KEIRIN_MONTHLY_SCHEDULE")
    month = data.get("month", {})
    cups = month.get("cups", [])
    venues = {str(v.get("id")): v for v in month.get("venues", [])}
    rows = []
    for cup in cups:
        venue = venues.get(str(cup.get("venueId")), {})
        slug = venue.get("slug") or venue.get("romaji") or venue.get("nameEn")
        # WINTICKET URLs use English slugs, but the schedule HTML contains the full cup link.
        rows.append({**cup, "venue_name": venue.get("name", ""), "source_month_url": url})

    links = re.findall(r'href="(/keirin/[^/]+/racecard/\d{10})"', html)
    cup_url_by_id = {}
    for href in links:
        cup_id = href.rstrip("/").split("/")[-1]
        cup_url_by_id[cup_id] = urljoin(BASE_URL, href)
    for row in rows:
        row["cup_url"] = cup_url_by_id.get(str(row.get("id")), "")
    return rows


def collect_cups(months_back=1, end_date=None):
    end = end_date or datetime.now(ZoneInfo("Asia/Tokyo")).date()
    end_month = datetime(end.year, end.month, 1)
    all_cups = []
    for offset in range(months_back):
        ym = add_months(end_month, -offset).strftime("%Y%m")
        cups = extract_month_cups(ym)
        all_cups.extend(cups)

    today_s = end.strftime("%Y%m%d")
    past = []
    for cup in all_cups:
        if not cup.get("cup_url"):
            continue
        if str(cup.get("endDate", "")) <= today_s:
            past.append(cup)
    return sorted(past, key=lambda x: (x.get("startDate", ""), x.get("id", "")))


def collect_race_urls_for_cup(cup):
    html = http_get(cup["cup_url"])
    state = extract_preloaded_state(html)
    data = find_query_data(state, "FETCH_KEIRIN_CUP")
    if not data:
        data = find_query_data(state, "FETCH_KEIRIN_CUP_RACES")
    schedules = {str(s.get("id")): s for s in data.get("schedules", [])}
    race_urls = []
    for race in data.get("races", []):
        schedule = schedules.get(str(race.get("scheduleId")), {})
        if not schedule:
            continue
        if int(race.get("status", 0) or 0) < 3:
            continue
        index = schedule.get("index")
        number = race.get("number")
        if not index or not number:
            continue
        race_urls.append(f"{cup['cup_url'].rstrip('/')}/{index}/{number}")
    return sorted(set(race_urls))


def parse_history_race(url):
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
    positions = {}
    for result in race_data.get("results", []) or []:
        player_id = str(result.get("playerId", ""))
        order = result.get("order")
        if player_id and isinstance(order, int) and order > 0:
            positions[player_id] = int(order)

    rows = []
    for entry in race_data.get("entries", []):
        if entry.get("absent"):
            continue
        player_id = str(entry.get("playerId", ""))
        finish_pos = positions.get(player_id)
        if not finish_pos:
            continue
        player = players.get(player_id, {})
        record = records.get(player_id, {})
        first_rate = pd.to_numeric(record.get("firstRate"), errors="coerce") / 100
        second_rate = pd.to_numeric(record.get("secondRate"), errors="coerce") / 100
        third_rate = pd.to_numeric(record.get("thirdRate"), errors="coerce") / 100
        history_row = {
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
            "finish_pos": finish_pos,
            "source_url": url,
            "player_name": player.get("name", ""),
            "race_class": race.get("class", ""),
            "race_type": race.get("raceType", ""),
            "distance": race.get("distance", np.nan),
        }
        rows.append(history_row)
    return rows


def build_history(months_back=1, max_cups=None, max_races=None, sleep_sec=0.2):
    ensure_dirs()
    cups = collect_cups(months_back=months_back)
    if max_cups:
        cups = cups[-max_cups:]
    if not cups:
        raise ValueError("no past cups found")

    failures = []
    all_race_urls = []
    for cup_i, cup in enumerate(cups, start=1):
        try:
            race_urls = collect_race_urls_for_cup(cup)
            all_race_urls.extend(race_urls)
            print(f"cup {cup_i}/{len(cups)}: {cup.get('cup_url')} races={len(race_urls)}")
        except Exception as e:
            failures.append({"cup": cup.get("id"), "url": cup.get("cup_url"), "error": str(e)})
            print(f"failed cup {cup_i}/{len(cups)} {cup.get('cup_url')} error={e}")

    all_race_urls = sorted(set(all_race_urls))
    if max_races and len(all_race_urls) > max_races:
        indexes = np.linspace(0, len(all_race_urls) - 1, max_races, dtype=int)
        race_urls_to_fetch = [all_race_urls[i] for i in indexes]
    else:
        race_urls_to_fetch = all_race_urls

    all_rows = []
    for race_count, race_url in enumerate(race_urls_to_fetch, start=1):
        try:
            rows = parse_history_race(race_url)
            all_rows.extend(rows)
            print(f"history race {race_count}/{len(race_urls_to_fetch)}: {race_url} rows={len(rows)}")
        except Exception as e:
            failures.append({"race_url": race_url, "error": str(e)})
            print(f"failed race {race_count}/{len(race_urls_to_fetch)}: {race_url} error={e}")
        time.sleep(sleep_sec)

    if not all_rows:
        raise ValueError(f"no history rows built: {failures[:3]}")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(["race_id", "player_id"]).sort_values(["date", "venue", "race_no", "car_no"])
    df.to_csv(HISTORY_CSV, index=False)

    metadata = {
        "source": "WINTICKET historical racecards",
        "built_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "months_back": months_back,
        "cups": len(cups),
        "candidate_races": len(all_race_urls),
        "fetched_races": len(race_urls_to_fetch),
        "races": int(df["race_id"].nunique()),
        "rows": int(len(df)),
        "failures": failures,
    }
    (RAW_DIR / "history_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months-back", type=int, default=1)
    parser.add_argument("--max-cups", type=int, default=None)
    parser.add_argument("--max-races", type=int, default=None)
    args = parser.parse_args()
    build_history(args.months_back, args.max_cups, args.max_races)


if __name__ == "__main__":
    main()
