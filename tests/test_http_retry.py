import unittest
from types import SimpleNamespace
from unittest import mock

import fetch_today_entries


class HttpRetryTests(unittest.TestCase):
    def test_http_get_retries_rate_limit_response(self):
        limited = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": "0"},
            raise_for_status=mock.Mock(),
            text="",
            encoding=None,
        )
        success = SimpleNamespace(
            status_code=200,
            headers={},
            raise_for_status=mock.Mock(),
            text="ok",
            encoding=None,
        )

        with mock.patch.object(
            fetch_today_entries.requests, "get", create=True, side_effect=[limited, success]
        ) as request, mock.patch.object(fetch_today_entries, "wait_for_http_slot"), mock.patch.object(
            fetch_today_entries.time, "sleep"
        ):
            result = fetch_today_entries.http_get("https://example.test", attempts=2)

        self.assertEqual(result, "ok")
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
