import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, HISTORY_TRIFECTA_ODDS_CSV, RAW_DIR, ensure_dirs
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


RACE_CACHE_DIR = RAW_DIR / "race_cache"
TRIFECTA_ODDS_COLUMNS = [
    "date",
    "venue",
    "race_no",
    "race_id",
    "buy",
    "trifecta_odds",
    "popularity_order",
    "is_actual",
    "actual_trifecta",
    "source_url",
]


def race_cache_path(url):
    race_id = url.rstrip("/").split("/")[-3:]
    safe = "_".join(race_id)
    return RACE_CACHE_DIR / f"{safe}.json"


def parse_history_race(url, use_cache=True):
    cache_path = race_cache_path(url)
    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        odds_rows = data.get("odds_rows", [])
        if rows and odds_rows:
            return rows, odds_rows

    html = http_get(url)
    state = extract_preloaded_state(html)
    race_data = find_query_data(state, "FETCH_KEIRIN_RACE")
    odds_data = find_query_data(state, "FETCH_KEIRIN_RACE_ODDS")
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

    odds_rows = []
    actual_top3 = []
    entry_by_player = {str(e.get("playerId")): int(e.get("number")) for e in race_data.get("entries", []) if e.get("number")}
    for result in sorted(race_data.get("results", []) or [], key=lambda x: x.get("order", 999)):
        if result.get("order") in [1, 2, 3]:
            car = entry_by_player.get(str(result.get("playerId")))
            if car:
                actual_top3.append(car)
    actual_trifecta = "-".join(str(x) for x in actual_top3[:3]) if len(actual_top3) >= 3 else ""

    trifecta_items = odds_data.get("trifecta", []) or race_data.get("trifecta", []) or []
    for item in trifecta_items:
        key = item.get("key", [])
        if len(key) != 3 or item.get("absent"):
            continue
        buy = f"{int(key[0])}-{int(key[1])}-{int(key[2])}"
        odds_rows.append(
            {
                "date": race_date,
                "venue": venue,
                "race_no": race_no,
                "race_id": race_id,
                "buy": buy,
                "trifecta_odds": item.get("odds", np.nan),
                "popularity_order": item.get("popularityOrder", np.nan),
                "is_actual": buy == actual_trifecta,
                "actual_trifecta": actual_trifecta,
                "source_url": url,
            }
        )

    if use_cache:
        RACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"rows": rows, "odds_rows": odds_rows}, ensure_ascii=False), encoding="utf-8")

    return rows, odds_rows


def collect_all_race_urls(cups, workers=1):
    failures = []
    all_race_urls = []
    workers = max(int(workers or 1), 1)
    if workers == 1:
        for cup_i, cup in enumerate(cups, start=1):
            try:
                race_urls = collect_race_urls_for_cup(cup)
                all_race_urls.extend(race_urls)
                print(f"cup {cup_i}/{len(cups)}: {cup.get('cup_url')} races={len(race_urls)}", flush=True)
            except Exception as e:
                failures.append({"cup": cup.get("id"), "url": cup.get("cup_url"), "error": str(e)})
                print(f"failed cup {cup_i}/{len(cups)} {cup.get('cup_url')} error={e}", flush=True)
        return sorted(set(all_race_urls)), failures

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(collect_race_urls_for_cup, cup): (cup_i, cup) for cup_i, cup in enumerate(cups, start=1)}
        done = 0
        for future in as_completed(future_map):
            cup_i, cup = future_map[future]
            done += 1
            try:
                race_urls = future.result()
                all_race_urls.extend(race_urls)
                print(
                    f"cup {done}/{len(cups)}: {cup.get('cup_url')} races={len(race_urls)} total={len(set(all_race_urls))}",
                    flush=True,
                )
            except Exception as e:
                failures.append({"cup": cup.get("id"), "url": cup.get("cup_url"), "error": str(e)})
                print(f"failed cup {cup_i}/{len(cups)} {cup.get('cup_url')} error={e}", flush=True)
    return sorted(set(all_race_urls)), failures


def fetch_history_races(race_urls, sleep_sec=0.2, use_cache=True, workers=1):
    failures = []
    all_rows = []
    all_odds_rows = []
    workers = max(int(workers or 1), 1)
    if workers == 1:
        for race_count, race_url in enumerate(race_urls, start=1):
            try:
                rows, odds_rows = parse_history_race(race_url, use_cache=use_cache)
                all_rows.extend(rows)
                all_odds_rows.extend(odds_rows)
                print(f"history race {race_count}/{len(race_urls)}: {race_url} rows={len(rows)} odds={len(odds_rows)}", flush=True)
            except Exception as e:
                failures.append({"race_url": race_url, "error": str(e)})
                print(f"failed race {race_count}/{len(race_urls)}: {race_url} error={e}", flush=True)
            time.sleep(sleep_sec)
        return all_rows, all_odds_rows, failures

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(parse_history_race, race_url, use_cache): race_url for race_url in race_urls}
        for race_count, future in enumerate(as_completed(future_map), start=1):
            race_url = future_map[future]
            try:
                rows, odds_rows = future.result()
                all_rows.extend(rows)
                all_odds_rows.extend(odds_rows)
                print(f"history race {race_count}/{len(race_urls)}: {race_url} rows={len(rows)} odds={len(odds_rows)}", flush=True)
            except Exception as e:
                failures.append({"race_url": race_url, "error": str(e)})
                print(f"failed race {race_count}/{len(race_urls)}: {race_url} error={e}", flush=True)
    return all_rows, all_odds_rows, failures


def build_history(months_back=1, max_cups=None, max_races=None, sleep_sec=0.2, use_cache=True, workers=1):
    ensure_dirs()
    cups = collect_cups(months_back=months_back)
    if max_cups:
        cups = cups[-max_cups:]
    if not cups:
        raise ValueError("no past cups found")

    all_race_urls, failures = collect_all_race_urls(cups, workers=workers)
    if max_races and len(all_race_urls) > max_races:
        indexes = np.linspace(0, len(all_race_urls) - 1, max_races, dtype=int)
        race_urls_to_fetch = [all_race_urls[i] for i in indexes]
    else:
        race_urls_to_fetch = all_race_urls

    all_rows, all_odds_rows, race_failures = fetch_history_races(race_urls_to_fetch, sleep_sec, use_cache, workers=workers)
    failures.extend(race_failures)

    if not all_rows:
        raise ValueError(f"no history rows built: {failures[:3]}")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(["race_id", "player_id"]).sort_values(["date", "venue", "race_no", "car_no"])
    df.to_csv(HISTORY_CSV, index=False)

    odds_df = pd.DataFrame(all_odds_rows, columns=TRIFECTA_ODDS_COLUMNS)
    if len(odds_df):
        odds_df = odds_df.drop_duplicates(["race_id", "buy"]).sort_values(["date", "venue", "race_no", "popularity_order", "buy"])
    odds_df.to_csv(HISTORY_TRIFECTA_ODDS_CSV, index=False)

    metadata = {
        "source": "WINTICKET historical racecards",
        "built_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "months_back": months_back,
        "cups": len(cups),
        "candidate_races": len(all_race_urls),
        "fetched_races": len(race_urls_to_fetch),
        "races": int(df["race_id"].nunique()),
        "rows": int(len(df)),
        "trifecta_odds_rows": int(len(odds_df)),
        "workers": int(workers or 1),
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
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    build_history(args.months_back, args.max_cups, args.max_races, args.sleep_sec, not args.no_cache, args.workers)


if __name__ == "__main__":
    main()
