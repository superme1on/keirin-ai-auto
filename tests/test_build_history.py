import unittest

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


if __name__ == "__main__":
    unittest.main()
