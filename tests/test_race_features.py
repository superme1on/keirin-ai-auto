import json
import unittest

from race_features import build_entry_rows, recent_avg_finish
from settle_results import settled_return_yen


class RaceFeatureTests(unittest.TestCase):
    def test_completed_race_keeps_non_finishers(self):
        race_id = "012320260710"
        race_data = {
            "race": {
                "id": race_id,
                "number": 1,
                "class": "A",
                "raceType": "test",
                "raceType3": "test",
                "entriesNumber": 4,
                "distance": 400,
                "weather": "\u96e8",
                "windSpeed": "1.5",
                "startAt": 1783650000,
                "closeAt": 1783649700,
            },
            "schedule": {"day": 1},
            "entries": [
                {"number": car, "playerId": str(car), "absent": False, "bracketNumber": car}
                for car in range(1, 5)
            ],
            "players": [{"id": str(car), "name": f"P{car}", "age": 30} for car in range(1, 5)],
            "records": [{"playerId": str(car), "racePoint": 80 - car, "firstRate": 10} for car in range(1, 5)],
            "results": [
                {"playerId": "1", "order": 1, "accidentName": ""},
                {"playerId": "2", "order": 0, "accident": "\u843d", "accidentName": "\u843d\u8eca"},
                {"playerId": "3", "order": 2, "accidentName": ""},
                {"playerId": "4", "order": 3, "accidentName": ""},
            ],
            "linePrediction": {
                "lineType": "test",
                "lines": [
                    {"entries": [{"numbers": [1]}, {"numbers": [2]}]},
                    {"entries": [{"numbers": [3]}, {"numbers": [4]}]},
                ],
            },
        }
        rows = build_entry_rows(race_data, "2026-07-10", "V", 1, race_id, "https://example.test", True)
        self.assertEqual(len(rows), 4)
        by_car = {row["car_no"]: row for row in rows}
        self.assertEqual(by_car[2]["finish_pos"], 5)
        self.assertEqual(by_car[2]["is_dnf"], 1)
        self.assertEqual(by_car[2]["result_status"], "\u843d\u8eca")
        self.assertEqual(by_car[2]["line_position"], 2)

    def test_recent_form_excludes_current_race(self):
        record = {
            "currentCupResults": [
                {"raceId": "012320260710", "order": 1},
                {"raceId": "012320260709", "order": 4},
            ]
        }
        self.assertEqual(recent_avg_finish(record, "2026-07-10", "012320260710"), 4.0)

    def test_settlement_uses_official_payoff(self):
        row = {
            "is_decided": True,
            "is_hit": True,
            "bet_type": "quinella_place",
            "buy": "1-4",
            "stake_yen": 200,
            "payouts_quinella_place_json": json.dumps({"1-4": 2150}),
        }
        self.assertEqual(settled_return_yen(row, fallback_return=3940), 4300)


if __name__ == "__main__":
    unittest.main()
