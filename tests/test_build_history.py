import unittest
from unittest import mock

import build_history
from build_history import extract_cup_url_by_id


class BuildHistoryTests(unittest.TestCase):
    def test_extracts_current_and_legacy_cup_links(self):
        html = """
        <a href="/keirin/aomori/racecard/2026082412/races">current</a>
        <a href="/keirin/nara/racecard/2026081453">legacy</a>
        """

        links = extract_cup_url_by_id(html)

        self.assertEqual(
            links["2026082412"],
            "https://www.winticket.jp/keirin/aomori/racecard/2026082412",
        )
        self.assertEqual(
            links["2026081453"],
            "https://www.winticket.jp/keirin/nara/racecard/2026081453",
        )

    def test_rejects_history_when_too_few_race_urls_are_resolved(self):
        cups = [{"cup_url": "https://example.test/cup"}]
        with mock.patch.object(build_history, "ensure_dirs"), mock.patch.object(
            build_history, "collect_cups", return_value=cups
        ), mock.patch.object(
            build_history, "collect_all_race_urls", return_value=(["race-1"], [{"error": "429"}])
        ), mock.patch.object(build_history, "fetch_history_races") as fetch_history:
            with self.assertRaisesRegex(ValueError, "coverage too small before race fetch"):
                build_history.build_history(min_races=2)

        fetch_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
