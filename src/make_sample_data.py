import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import ensure_dirs, HISTORY_CSV, TODAY_CSV

VENUES = ["立川", "京王閣", "松戸", "川崎", "小田原", "平塚", "大宮", "宇都宮", "名古屋", "岸和田"]
STYLES = ["逃", "捲", "差", "追", "両"]


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def make_history(n_races=1200, riders_per_race=7, seed=42):
    rng = np.random.default_rng(seed)
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    rows = []

    player_pool = np.arange(10001, 10551)

    for i in range(n_races):
        race_date = today - timedelta(days=int(n_races / 8) - int(i / 8))
        venue = rng.choice(VENUES)
        race_no = int(i % 12) + 1
        race_id = f"{race_date.strftime('%Y%m%d')}{VENUES.index(venue)+1:02d}{race_no:02d}{i:04d}"

        riders = rng.choice(player_pool, size=riders_per_race, replace=False)
        base_skill = rng.normal(0, 1, riders_per_race)

        scores = 83 + base_skill * 4 + rng.normal(0, 2, riders_per_race)
        win_rates = np.clip(0.12 + base_skill * 0.04 + rng.normal(0, 0.03, riders_per_race), 0.01, 0.45)
        place2_rates = np.clip(win_rates + 0.12 + rng.normal(0, 0.04, riders_per_race), 0.03, 0.7)
        place3_rates = np.clip(place2_rates + 0.15 + rng.normal(0, 0.05, riders_per_race), 0.05, 0.9)
        back_counts = np.clip(rng.poisson(3 + np.maximum(base_skill, 0)), 0, 20)
        recent_avg_finish = np.clip(4.5 - base_skill * 0.6 + rng.normal(0, 0.6, riders_per_race), 1, 7)
        days_since = rng.integers(3, 40, riders_per_race)
        venue_win_rate = np.clip(win_rates + rng.normal(0, 0.04, riders_per_race), 0.01, 0.5)

        ability = (
            scores * 0.06
            + win_rates * 8
            + place2_rates * 3
            + venue_win_rate * 4
            - recent_avg_finish * 0.35
            + rng.normal(0, 0.5, riders_per_race)
        )

        performance = ability + rng.normal(0, 1.2, riders_per_race)
        order_idx = np.argsort(-performance)
        finish_pos = np.empty(riders_per_race, dtype=int)
        finish_pos[order_idx] = np.arange(1, riders_per_race + 1)

        probs = softmax(ability)
        odds = np.clip(0.78 / probs + rng.normal(0, 0.8, riders_per_race), 1.1, 80)

        for j in range(riders_per_race):
            rows.append({
                "race_id": race_id,
                "date": str(race_date),
                "venue": venue,
                "race_no": race_no,
                "player_id": int(riders[j]),
                "car_no": j + 1,
                "age": int(rng.integers(22, 48)),
                "score": round(float(scores[j]), 2),
                "win_rate": round(float(win_rates[j]), 3),
                "place2_rate": round(float(place2_rates[j]), 3),
                "place3_rate": round(float(place3_rates[j]), 3),
                "back_count": int(back_counts[j]),
                "style": str(rng.choice(STYLES)),
                "recent_avg_finish": round(float(recent_avg_finish[j]), 2),
                "days_since_last_race": int(days_since[j]),
                "venue_win_rate": round(float(venue_win_rate[j]), 3),
                "odds_win": round(float(odds[j]), 1),
                "finish_pos": int(finish_pos[j]),
            })

    return pd.DataFrame(rows)


def make_today(n_races=12, riders_per_race=7, seed=2026):
    rng = np.random.default_rng(seed)
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    rows = []
    player_pool = np.arange(20001, 20601)

    for i in range(n_races):
        venue = rng.choice(VENUES)
        race_no = i + 1
        race_id = f"{today.strftime('%Y%m%d')}{VENUES.index(venue)+1:02d}{race_no:02d}"
        riders = rng.choice(player_pool, size=riders_per_race, replace=False)
        base_skill = rng.normal(0, 1, riders_per_race)

        scores = 83 + base_skill * 4 + rng.normal(0, 2, riders_per_race)
        win_rates = np.clip(0.12 + base_skill * 0.04 + rng.normal(0, 0.03, riders_per_race), 0.01, 0.45)
        place2_rates = np.clip(win_rates + 0.12 + rng.normal(0, 0.04, riders_per_race), 0.03, 0.7)
        place3_rates = np.clip(place2_rates + 0.15 + rng.normal(0, 0.05, riders_per_race), 0.05, 0.9)
        back_counts = np.clip(rng.poisson(3 + np.maximum(base_skill, 0)), 0, 20)
        recent_avg_finish = np.clip(4.5 - base_skill * 0.6 + rng.normal(0, 0.6, riders_per_race), 1, 7)
        days_since = rng.integers(3, 40, riders_per_race)
        venue_win_rate = np.clip(win_rates + rng.normal(0, 0.04, riders_per_race), 0.01, 0.5)

        ability = scores * 0.06 + win_rates * 8 + place2_rates * 3 + venue_win_rate * 4 - recent_avg_finish * 0.35
        probs = softmax(ability)
        odds = np.clip(0.78 / probs + rng.normal(0, 0.8, riders_per_race), 1.1, 80)

        for j in range(riders_per_race):
            rows.append({
                "race_id": race_id,
                "date": str(today),
                "venue": venue,
                "race_no": race_no,
                "player_id": int(riders[j]),
                "car_no": j + 1,
                "age": int(rng.integers(22, 48)),
                "score": round(float(scores[j]), 2),
                "win_rate": round(float(win_rates[j]), 3),
                "place2_rate": round(float(place2_rates[j]), 3),
                "place3_rate": round(float(place3_rates[j]), 3),
                "back_count": int(back_counts[j]),
                "style": str(rng.choice(STYLES)),
                "recent_avg_finish": round(float(recent_avg_finish[j]), 2),
                "days_since_last_race": int(days_since[j]),
                "venue_win_rate": round(float(venue_win_rate[j]), 3),
                "odds_win": round(float(odds[j]), 1),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    if args.if_missing and HISTORY_CSV.exists() and TODAY_CSV.exists():
        print("sample data already exists; skip")
        return

    if not HISTORY_CSV.exists():
        hist = make_history()
        hist.to_csv(HISTORY_CSV, index=False)
        print(f"created {HISTORY_CSV} rows={len(hist)}")
    else:
        print(f"exists {HISTORY_CSV}")

    if not TODAY_CSV.exists():
        today = make_today()
        today.to_csv(TODAY_CSV, index=False)
        print(f"created {TODAY_CSV} rows={len(today)}")
    else:
        print(f"exists {TODAY_CSV}")


if __name__ == "__main__":
    main()
