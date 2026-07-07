import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_history import (
    ODDS_COLUMNS,
    TRIFECTA_ODDS_COLUMNS,
    collect_all_race_urls,
    collect_cups,
    fetch_history_races,
)
from common import HISTORY_CSV, HISTORY_ODDS_CSV, HISTORY_TRIFECTA_ODDS_CSV, RAW_DIR, ensure_dirs


def existing_race_ids():
    if not HISTORY_CSV.exists() or HISTORY_CSV.stat().st_size == 0:
        return set()
    return set(pd.read_csv(HISTORY_CSV, dtype={"race_id": str}, usecols=["race_id"])["race_id"].dropna().astype(str))


def append_csv(df, path: Path):
    if len(df) == 0:
        return
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def select_urls(urls, sample_mod=None, sample_offset=0, max_races=None):
    selected = list(urls)
    if sample_mod:
        sample_mod = int(sample_mod)
        sample_offset = int(sample_offset)
        if sample_mod <= 0:
            raise ValueError("--sample-mod must be positive")
        if sample_offset < 0 or sample_offset >= sample_mod:
            raise ValueError("--sample-offset must be between 0 and sample_mod - 1")
        selected = selected[sample_offset::sample_mod]
    elif max_races and len(selected) > max_races:
        indexes = np.linspace(0, len(selected) - 1, int(max_races), dtype=int)
        selected = [selected[i] for i in indexes]
    if sample_mod and max_races:
        selected = selected[: int(max_races)]
    return selected


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def append_history_rows(rows, odds_rows, known_ids):
    hist = pd.DataFrame(rows)
    if len(hist) == 0:
        return 0, 0
    hist["race_id"] = hist["race_id"].astype(str)
    hist = hist[~hist["race_id"].isin(known_ids)].copy()
    if len(hist) == 0:
        return 0, 0

    new_ids = set(hist["race_id"].astype(str))
    known_ids.update(new_ids)
    hist = hist.drop_duplicates(["race_id", "player_id"]).sort_values(["date", "venue", "race_no", "car_no"])
    append_csv(hist, HISTORY_CSV)

    odds = pd.DataFrame(odds_rows, columns=ODDS_COLUMNS)
    if len(odds):
        odds["race_id"] = odds["race_id"].astype(str)
        odds = odds[odds["race_id"].isin(new_ids)].copy()
        odds = odds.drop_duplicates(["race_id", "bet_type", "buy"]).sort_values(
            ["date", "venue", "race_no", "bet_type", "popularity_order", "buy"]
        )
        append_csv(odds, HISTORY_ODDS_CSV)

        trifecta = odds[odds["bet_type"].eq("trifecta")].copy()
        if len(trifecta):
            trifecta["trifecta_odds"] = trifecta["odds_used"]
            trifecta["actual_trifecta"] = trifecta["actual_buy"]
            append_csv(trifecta[TRIFECTA_ODDS_COLUMNS], HISTORY_TRIFECTA_ODDS_CSV)

    return int(hist["race_id"].nunique()), int(len(odds)) if len(odds) else 0


def write_metadata(metadata):
    path = RAW_DIR / "history_expand_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def expand_history(
    months_back=60,
    sample_mod=5,
    sample_offset=0,
    max_races=None,
    chunk_size=1000,
    workers=8,
    sleep_sec=0.05,
    progress_every=100,
    no_cache=False,
):
    ensure_dirs()
    cups = collect_cups(months_back=months_back)
    urls, cup_failures = collect_all_race_urls(cups, workers=workers, progress_every=max(progress_every, 1))
    selected = select_urls(urls, sample_mod=sample_mod, sample_offset=sample_offset, max_races=max_races)
    known_ids = existing_race_ids()

    fetched_races = 0
    appended_races = 0
    appended_odds_rows = 0
    race_failures = []

    for chunk_i, urls_chunk in enumerate(chunked(selected, chunk_size), start=1):
        print(
            f"expand chunk {chunk_i}: urls={len(urls_chunk)} fetched={fetched_races}/{len(selected)} "
            f"appended_races={appended_races}",
            flush=True,
        )
        rows, odds_rows, failures = fetch_history_races(
            urls_chunk,
            sleep_sec=sleep_sec,
            use_cache=not no_cache,
            workers=workers,
            progress_every=progress_every,
        )
        fetched_races += len(urls_chunk)
        race_failures.extend(failures)
        new_races, new_odds = append_history_rows(rows, odds_rows, known_ids)
        appended_races += new_races
        appended_odds_rows += new_odds
        write_metadata(
            {
                "source": "WINTICKET historical racecards incremental expansion",
                "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
                "months_back": months_back,
                "cups": len(cups),
                "candidate_races": len(urls),
                "selected_races": len(selected),
                "fetched_races": fetched_races,
                "appended_races": appended_races,
                "appended_odds_rows": appended_odds_rows,
                "sample_mod": sample_mod,
                "sample_offset": sample_offset,
                "max_races": max_races,
                "chunk_size": chunk_size,
                "workers": workers,
                "cup_failures": cup_failures,
                "race_failures": race_failures,
            }
        )

    return appended_races


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months-back", type=int, default=60)
    parser.add_argument("--sample-mod", type=int, default=5)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    expand_history(
        months_back=args.months_back,
        sample_mod=args.sample_mod,
        sample_offset=args.sample_offset,
        max_races=args.max_races,
        chunk_size=args.chunk_size,
        workers=args.workers,
        sleep_sec=args.sleep_sec,
        progress_every=args.progress_every,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
