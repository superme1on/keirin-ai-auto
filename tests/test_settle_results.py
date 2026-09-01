import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import settle_results


class SettleResultsTests(unittest.TestCase):
    def test_empty_bet_file_writes_zero_summary_without_fetching_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            latest_bets = output_dir / "latest_bets.csv"
            settled_bets = output_dir / "settled_bets.csv"
            purchase_plan = output_dir / "purchase_plan.csv"
            summary_csv = output_dir / "settlement_summary.csv"
            report_md = output_dir / "japanese_report.md"
            pd.DataFrame(
                columns=["date", "venue", "race_no", "race_id", "bet_type", "buy"]
            ).to_csv(latest_bets, index=False)

            with mock.patch.multiple(
                settle_results,
                LATEST_BETS_CSV=latest_bets,
                SETTLED_BETS_CSV=settled_bets,
                PURCHASE_PLAN_CSV=purchase_plan,
                SETTLEMENT_SUMMARY_CSV=summary_csv,
                REPORT_MD=report_md,
            ), mock.patch.object(settle_results, "ensure_dirs"), mock.patch.object(
                settle_results, "fetch_results"
            ) as fetch_results:
                settle_results.run_settlement(argparse.Namespace(min_expected_profit=0.0))

            fetch_results.assert_not_called()
            summary = pd.read_csv(summary_csv)
            self.assertEqual(summary["bets"].sum(), 0)
            self.assertEqual(summary["stake_yen"].sum(), 0)
            self.assertTrue(settled_bets.exists())
            self.assertTrue(purchase_plan.exists())
            self.assertTrue(report_md.exists())


if __name__ == "__main__":
    unittest.main()
