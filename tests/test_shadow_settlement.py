import json
import unittest

import pandas as pd

from settle_shadow import parse_race_id, settle_shadow_rows, summarize_rows


class ShadowSettlementTests(unittest.TestCase):
    def test_parse_race_id(self):
        parsed = parse_race_id("051320260711")
        self.assertEqual(parsed["race_no"], 5)
        self.assertEqual(parsed["venue_id"], "13")
        self.assertEqual(parsed["date"], "20260711")

    def test_official_payoff_is_used_in_shadow_summary(self):
        bets = pd.DataFrame(
            [
                {
                    "date": "2026-07-11",
                    "venue": "いわき平",
                    "race_no": 5,
                    "race_id": "051320260711",
                    "bet_type": "trio",
                    "bet_label": "3連複",
                    "buy": "2-3-4",
                    "stake_yen": 300,
                    "return_if_hit_yen": 99999,
                    "prediction_created_at_jst": "2026-07-11T10:00:00+09:00",
                }
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "race_id": "051320260711",
                    "actual_trifecta": "2-3-4",
                    "actual_trio": "2-3-4",
                    "payouts_trio_json": json.dumps({"2-3-4": 2150}),
                    "close_at": 1783735200,
                    "source_url": "https://example.test/race",
                }
            ]
        )
        settled = settle_shadow_rows(bets, results)
        self.assertTrue(bool(settled.iloc[0]["is_hit"]))
        self.assertEqual(settled.iloc[0]["actual_return_yen"], 6450)
        summary = summarize_rows(settled)
        self.assertEqual(summary["profit_yen"], 6150)
        self.assertEqual(summary["roi"], 20.5)


if __name__ == "__main__":
    unittest.main()
