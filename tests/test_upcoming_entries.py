import unittest

import pandas as pd

from fetch_upcoming_entries import select_upcoming_races


class UpcomingEntryTests(unittest.TestCase):
    def test_selects_only_near_close_window(self):
        schedule = pd.DataFrame(
            [
                {"race_id": "too_soon", "close_at": 1299},
                {"race_id": "window_start", "close_at": 1300},
                {"race_id": "window_end", "close_at": 3400},
                {"race_id": "too_late", "close_at": 3401},
            ]
        )
        selected = select_upcoming_races(schedule, now_epoch=1000, min_minutes=5, max_minutes=40)
        self.assertEqual(selected["race_id"].tolist(), ["window_start", "window_end"])


if __name__ == "__main__":
    unittest.main()
