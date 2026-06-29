import os
import sys
from pathlib import Path

import requests

from common import ensure_dirs, HISTORY_CSV, TODAY_CSV
from fetch_today_entries import fetch_today_entries


def download_csv(url: str, path: Path):
    print(f"download: {url} -> {path}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"saved: {path} bytes={path.stat().st_size}")


def main():
    ensure_dirs()

    history_url = os.getenv("KEIRIN_HISTORY_CSV_URL", "").strip()
    today_url = os.getenv("KEIRIN_TODAY_CSV_URL", "").strip()

    if history_url:
        download_csv(history_url, HISTORY_CSV)
    else:
        print("KEIRIN_HISTORY_CSV_URL is empty; skip history download")

    if today_url:
        download_csv(today_url, TODAY_CSV)
    else:
        auto_fetch = os.getenv("AUTO_FETCH_TODAY_ENTRIES", "1").strip().lower()
        if auto_fetch in {"1", "true", "yes", "on"}:
            print("KEIRIN_TODAY_CSV_URL is empty; fetch today entries from WINTICKET")
            fetch_today_entries()
        else:
            print("KEIRIN_TODAY_CSV_URL is empty; skip today download")

    print("ingest done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ingest failed: {e}", file=sys.stderr)
        raise
