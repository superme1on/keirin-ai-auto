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

from common import HISTORY_CSV, HISTORY_ODDS_CSV, HISTORY_TRIFECTA_ODDS_CSV, RAW_DIR, ensure_dirs
from fetch_today_entries import (
    BASE_URL,
    extract_preloaded_state,
    find_query_data,
    get_venue_name,
    http_get,
    normalize_date,
)
from race_features import build_entry_rows


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
CACHE_VERSION = 3
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
    "is_actual",
    "actual_buy",
    "source_url",
]
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
BET_SPECS = {
    "trifecta": {"source": "trifecta", "key_len": 3, "ordered": True},
    "trio": {"source": "trio", "key_len": 3, "ordered": False},
    "exacta": {"source": "exacta", "key_len": 2, "ordered": True},
    "quinella": {"source": "quinella", "key_len": 2, "ordered": False},
    "quinella_place": {"source": "quinellaPlace", "key_len": 2, "ordered": False, "actual_kind": "top3_pairs"},
    "bracket_exacta": {"source": "bracketExacta", "key_len": 2, "ordered": True, "key_kind": "bracket"},
    "bracket_quinella": {"source": "bracketQuinella", "key_len": 2, "ordered": False, "key_kind": "bracket"},
}


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
        if data.get("cache_version") == CACHE_VERSION and rows and odds_rows:
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

    rows = build_entry_rows(
        race_data,
        race_date=race_date,
        venue=venue,
        race_no=race_no,
        race_id=race_id,
        source_url=url,
        include_results=True,
    )

    entry_by_player = {str(e.get("playerId")): int(e.get("number")) for e in race_data.get("entries", []) if e.get("number")}
    bracket_by_car = {
        int(e.get("number")): int(e.get("bracketNumber"))
        for e in race_data.get("entries", [])
        if e.get("number") and e.get("bracketNumber")
    }
    actual_top3 = []
    for result in sorted(race_data.get("results", []) or [], key=lambda x: x.get("order", 999)):
        if result.get("order") in [1, 2, 3]:
            car = entry_by_player.get(str(result.get("playerId")))
            if car:
                actual_top3.append(car)
    odds_rows = build_odds_rows(odds_data or race_data, race_date, venue, race_no, race_id, actual_top3, bracket_by_car, url)

    if use_cache:
        RACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"cache_version": CACHE_VERSION, "rows": rows, "odds_rows": odds_rows}, ensure_ascii=False),
            encoding="utf-8",
        )

    return rows, odds_rows


def key_to_buy(key, ordered=True):
    values = [int(x) for x in key]
    if not ordered:
        values = sorted(values)
    return "-".join(str(x) for x in values)


def actual_buy_sets(actual_top3, bracket_by_car):
    cars = [int(x) for x in actual_top3[:3]]
    top2 = cars[:2]
    brackets = [bracket_by_car.get(car) for car in cars]
    bracket_top2 = [b for b in brackets[:2] if b is not None]
    actual = {
        "trifecta": {key_to_buy(cars, ordered=True)} if len(cars) == 3 else set(),
        "trio": {key_to_buy(cars, ordered=False)} if len(cars) == 3 else set(),
        "exacta": {key_to_buy(top2, ordered=True)} if len(top2) == 2 else set(),
        "quinella": {key_to_buy(top2, ordered=False)} if len(top2) == 2 else set(),
        "bracket_exacta": {key_to_buy(bracket_top2, ordered=True)} if len(bracket_top2) == 2 else set(),
        "bracket_quinella": {key_to_buy(bracket_top2, ordered=False)} if len(bracket_top2) == 2 else set(),
    }
    if len(cars) == 3:
        actual["quinella_place"] = {
            key_to_buy([cars[0], cars[1]], ordered=False),
            key_to_buy([cars[0], cars[2]], ordered=False),
            key_to_buy([cars[1], cars[2]], ordered=False),
        }
    else:
        actual["quinella_place"] = set()
    return actual


def build_odds_rows(odds_data, race_date, venue, race_no, race_id, actual_top3, bracket_by_car, url):
    actual = actual_buy_sets(actual_top3, bracket_by_car)
    rows = []
    for bet_type, spec in BET_SPECS.items():
        items = odds_data.get(spec["source"], []) or []
        for item in items:
            key = item.get("key", [])
            if len(key) != spec["key_len"] or item.get("absent"):
                continue
            buy = key_to_buy(key, ordered=spec.get("ordered", True))
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
                    "buy": buy,
                    "odds": odds,
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                    "odds_used": odds_used,
                    "popularity_order": item.get("popularityOrder", np.nan),
                    "is_actual": buy in actual.get(bet_type, set()),
                    "actual_buy": "|".join(sorted(actual.get(bet_type, set()))),
                    "source_url": url,
                }
            )
    return rows


def collect_all_race_urls(cups, workers=1, progress_every=1):
    failures = []
    all_race_urls = []
    workers = max(int(workers or 1), 1)
    if workers == 1:
        for cup_i, cup in enumerate(cups, start=1):
            try:
                race_urls = collect_race_urls_for_cup(cup)
                all_race_urls.extend(race_urls)
                if cup_i % progress_every == 0 or cup_i == len(cups):
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
                if done % progress_every == 0 or done == len(cups):
                    print(
                        f"cup {done}/{len(cups)}: {cup.get('cup_url')} races={len(race_urls)} total={len(set(all_race_urls))}",
                        flush=True,
                    )
            except Exception as e:
                failures.append({"cup": cup.get("id"), "url": cup.get("cup_url"), "error": str(e)})
                print(f"failed cup {cup_i}/{len(cups)} {cup.get('cup_url')} error={e}", flush=True)
    return sorted(set(all_race_urls)), failures


def fetch_history_races(race_urls, sleep_sec=0.2, use_cache=True, workers=1, progress_every=1):
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
                if race_count % progress_every == 0 or race_count == len(race_urls):
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
                if race_count % progress_every == 0 or race_count == len(race_urls):
                    print(f"history race {race_count}/{len(race_urls)}: {race_url} rows={len(rows)} odds={len(odds_rows)}", flush=True)
            except Exception as e:
                failures.append({"race_url": race_url, "error": str(e)})
                print(f"failed race {race_count}/{len(race_urls)}: {race_url} error={e}", flush=True)
    return all_rows, all_odds_rows, failures


def build_history(
    months_back=1,
    max_cups=None,
    max_races=None,
    sleep_sec=0.2,
    use_cache=True,
    workers=1,
    progress_every=1,
    sample_mod=None,
    sample_offset=0,
):
    ensure_dirs()
    cups = collect_cups(months_back=months_back)
    if max_cups:
        cups = cups[-max_cups:]
    if not cups:
        raise ValueError("no past cups found")

    all_race_urls, failures = collect_all_race_urls(cups, workers=workers, progress_every=progress_every)
    if sample_mod:
        sample_mod = int(sample_mod)
        sample_offset = int(sample_offset)
        if sample_mod <= 0:
            raise ValueError("--sample-mod must be positive")
        if sample_offset < 0 or sample_offset >= sample_mod:
            raise ValueError("--sample-offset must be between 0 and sample_mod - 1")
        race_urls_to_fetch = all_race_urls[sample_offset::sample_mod]
        if max_races:
            race_urls_to_fetch = race_urls_to_fetch[:max_races]
    elif max_races and len(all_race_urls) > max_races:
        indexes = np.linspace(0, len(all_race_urls) - 1, max_races, dtype=int)
        race_urls_to_fetch = [all_race_urls[i] for i in indexes]
    else:
        race_urls_to_fetch = all_race_urls

    all_rows, all_odds_rows, race_failures = fetch_history_races(
        race_urls_to_fetch, sleep_sec, use_cache, workers=workers, progress_every=progress_every
    )
    failures.extend(race_failures)

    if not all_rows:
        raise ValueError(f"no history rows built: {failures[:3]}")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(["race_id", "player_id"]).sort_values(["date", "venue", "race_no", "car_no"])
    df.to_csv(HISTORY_CSV, index=False)

    odds_df = pd.DataFrame(all_odds_rows, columns=ODDS_COLUMNS)
    if len(odds_df):
        odds_df = odds_df.drop_duplicates(["race_id", "bet_type", "buy"]).sort_values(
            ["date", "venue", "race_no", "bet_type", "popularity_order", "buy"]
        )
    odds_df.to_csv(HISTORY_ODDS_CSV, index=False)

    trifecta_df = odds_df[odds_df["bet_type"].eq("trifecta")].copy() if len(odds_df) else pd.DataFrame(columns=ODDS_COLUMNS)
    if len(trifecta_df):
        trifecta_df["trifecta_odds"] = trifecta_df["odds_used"]
        trifecta_df["actual_trifecta"] = trifecta_df["actual_buy"]
        trifecta_df = trifecta_df[TRIFECTA_ODDS_COLUMNS]
    else:
        trifecta_df = pd.DataFrame(columns=TRIFECTA_ODDS_COLUMNS)
    trifecta_df.to_csv(HISTORY_TRIFECTA_ODDS_CSV, index=False)

    metadata = {
        "source": "WINTICKET historical racecards",
        "built_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "months_back": months_back,
        "cups": len(cups),
        "candidate_races": len(all_race_urls),
        "fetched_races": len(race_urls_to_fetch),
        "sample_mod": sample_mod,
        "sample_offset": sample_offset if sample_mod else None,
        "races": int(df["race_id"].nunique()),
        "rows": int(len(df)),
        "odds_rows": int(len(odds_df)),
        "odds_races": int(odds_df["race_id"].nunique()) if len(odds_df) else 0,
        "bet_types": sorted(odds_df["bet_type"].dropna().unique().tolist()) if len(odds_df) else [],
        "trifecta_odds_rows": int(len(trifecta_df)),
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
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--sample-mod", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args()
    build_history(
        args.months_back,
        args.max_cups,
        args.max_races,
        args.sleep_sec,
        not args.no_cache,
        args.workers,
        args.progress_every,
        args.sample_mod,
        args.sample_offset,
    )


if __name__ == "__main__":
    main()
