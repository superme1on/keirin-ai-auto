import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from common import HISTORY_CSV, HISTORY_ODDS_CSV, OUTPUT_DIR, PROCESSED_DIR, ensure_dirs


INTEGRITY_CSV = PROCESSED_DIR / "race_integrity.csv"
REPORT_JSON = OUTPUT_DIR / "history_integrity_report.json"


def infer_starters(chunksize=500_000):
    cars_by_race = {}
    source_by_race = {}
    columns = ["race_id", "bet_type", "buy", "source_url"]
    for chunk in pd.read_csv(
        HISTORY_ODDS_CSV,
        dtype={"race_id": str, "bet_type": str, "buy": str},
        usecols=columns,
        chunksize=chunksize,
    ):
        exacta = chunk[chunk["bet_type"].eq("exacta")]
        for race_id, buy, source_url in exacta[["race_id", "buy", "source_url"]].itertuples(index=False):
            cars = cars_by_race.setdefault(str(race_id), set())
            for value in str(buy).split("-"):
                try:
                    cars.add(int(value))
                except ValueError:
                    continue
            if source_url and str(race_id) not in source_by_race:
                source_by_race[str(race_id)] = str(source_url)
    return cars_by_race, source_by_race


def audit_history(chunksize=500_000):
    ensure_dirs()
    history = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    history_counts = history.groupby("race_id")["car_no"].nunique().rename("history_entries")
    cars_by_race, source_by_race = infer_starters(chunksize=chunksize)
    starter_counts = pd.Series(
        {race_id: len(cars) for race_id, cars in cars_by_race.items()},
        name="listed_entries",
        dtype="int64",
    )

    integrity = pd.concat([history_counts, starter_counts], axis=1)
    integrity["missing_entries"] = integrity["listed_entries"] - integrity["history_entries"]
    integrity["is_complete"] = integrity["missing_entries"].eq(0)
    integrity["source_url"] = integrity.index.to_series().map(source_by_race)
    integrity = integrity.reset_index().rename(columns={"index": "race_id"})
    integrity.to_csv(INTEGRITY_CSV, index=False)

    comparable = integrity.dropna(subset=["history_entries", "listed_entries"])
    broken = comparable[~comparable["is_complete"]]
    report = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "history_rows": int(len(history)),
        "history_races": int(history["race_id"].nunique()),
        "comparable_races": int(len(comparable)),
        "complete_races": int(comparable["is_complete"].sum()),
        "broken_races": int(len(broken)),
        "broken_rate": float(len(broken) / len(comparable)) if len(comparable) else None,
        "missing_entry_rows": int(broken["missing_entries"].sum()),
        "integrity_csv": str(INTEGRITY_CSV),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report, integrity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--fail-on-leakage", action="store_true")
    args = parser.parse_args()
    report, _ = audit_history(chunksize=args.chunksize)
    if args.fail_on_leakage and report["broken_races"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
