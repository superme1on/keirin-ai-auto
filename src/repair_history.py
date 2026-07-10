import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from audit_history import INTEGRITY_CSV, audit_history
from build_history import fetch_history_races
from common import HISTORY_CSV, RAW_DIR, ensure_dirs


METADATA_JSON = RAW_DIR / "history_repair_metadata.json"


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def repair_history(enrich_all=False, workers=12, chunk_size=100, progress_every=25):
    ensure_dirs()
    if not INTEGRITY_CSV.exists():
        audit_history()
    integrity = pd.read_csv(INTEGRITY_CSV, dtype={"race_id": str})
    target = integrity if enrich_all else integrity[~integrity["is_complete"].astype(bool)]
    target = target.dropna(subset=["source_url"]).drop_duplicates("race_id")
    urls = target["source_url"].astype(str).tolist()
    expected = target.set_index("race_id")["listed_entries"].to_dict()

    replacement_rows = []
    failures = []
    fetched = 0
    for batch in chunked(urls, chunk_size):
        rows, _, batch_failures = fetch_history_races(
            batch,
            sleep_sec=0,
            use_cache=True,
            workers=workers,
            progress_every=progress_every,
        )
        fetched += len(batch)
        failures.extend(batch_failures)
        by_race = {}
        for row in rows:
            by_race.setdefault(str(row.get("race_id")), []).append(row)
        for race_id, race_rows in by_race.items():
            expected_count = int(expected.get(race_id, len(race_rows)))
            actual_count = len({int(row["car_no"]) for row in race_rows})
            if actual_count != expected_count:
                failures.append(
                    {
                        "race_id": race_id,
                        "error": f"entry count mismatch expected={expected_count} actual={actual_count}",
                    }
                )
                continue
            replacement_rows.extend(race_rows)
        print(
            f"repair progress: fetched={fetched}/{len(urls)} valid_rows={len(replacement_rows)} "
            f"failures={len(failures)}",
            flush=True,
        )

    replacement = pd.DataFrame(replacement_rows)
    if len(replacement):
        replacement["race_id"] = replacement["race_id"].astype(str)
        replacement_ids = set(replacement["race_id"])
        history = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
        history = history[~history["race_id"].isin(replacement_ids)]
        history = pd.concat([history, replacement], ignore_index=True, sort=False)
        history = history.drop_duplicates(["race_id", "player_id"], keep="last")
        history = history.sort_values(["date", "venue", "race_no", "car_no"], kind="mergesort")
        temp_path = HISTORY_CSV.with_suffix(".repairing.csv")
        history.to_csv(temp_path, index=False)
        os.replace(temp_path, HISTORY_CSV)
    else:
        replacement_ids = set()

    metadata = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "mode": "enrich_all" if enrich_all else "repair_broken_only",
        "target_races": int(len(urls)),
        "replaced_races": int(len(replacement_ids)),
        "replacement_rows": int(len(replacement)),
        "workers": int(workers),
        "failures": failures,
    }
    METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich-all", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    repair_history(
        enrich_all=args.enrich_all,
        workers=args.workers,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
