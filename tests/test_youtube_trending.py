"""Tests for youtube_trending module."""
import json
import sys
import types
import unittest
from unittest.mock import patch

# Provide minimal stubs if optional runtime deps are absent in bare Python env.
if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")
    sys.modules["httpx"] = httpx_stub

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _DummyRouter:
        def get(self, *args, **kwargs):
            return lambda fn: fn

    fastapi_stub.APIRouter = _DummyRouter
    fastapi_stub.Query = lambda default=None, **kwargs: default
    sys.modules["fastapi"] = fastapi_stub

if "security_utils" not in sys.modules:
    sec_stub = types.ModuleType("security_utils")
    sec_stub._ip_is_public = lambda ip: True
    sec_stub.assert_public_url = lambda url: url
    sys.modules["security_utils"] = sec_stub

import youtube_trending as yt  # noqa: E402


class TestYouTubeTrendingHelpers(unittest.TestCase):
    def setUp(self):
        yt._cache.clear()

    def test_parse_duration_seconds(self):
        self.assertEqual(yt.parse_duration_seconds("12:34"), 754)
        self.assertEqual(yt.parse_duration_seconds("1:02:03"), 3723)
        self.assertEqual(yt.parse_duration_seconds("0:45"), 45)
        self.assertEqual(yt.parse_duration_seconds("PT12M34S"), 754)
        self.assertEqual(yt.parse_duration_seconds("PT1H2M3S"), 3723)
        self.assertIsNone(yt.parse_duration_seconds(""))
        self.assertIsNone(yt.parse_duration_seconds("LIVE"))

    def test_parse_view_count(self):
        self.assertEqual(yt.parse_view_count("1.2M views"), 1_200_000)
        self.assertEqual(yt.parse_view_count("450K views"), 450_000)
        self.assertEqual(yt.parse_view_count("12,345 views"), 12_345)
        self.assertEqual(yt.parse_view_count("2.5B views"), 2_500_000_000)
        self.assertIsNone(yt.parse_view_count("No views"))

    def test_extract_and_collect_yt_initial_data(self):
        sample_data = {
            "contents": {
                "twoColumnBrowseResultsRenderer": {
                    "tabs": [
                        {
                            "tabRenderer": {
                                "content": {
                                    "sectionListRenderer": {
                                        "contents": [
                                            {
                                                "videoRenderer": {
                                                    "videoId": "abc123XYZ00",
                                                    "title": {"runs": [{"text": "Trending Video"}]},
                                                    "ownerText": {
                                                        "runs": [
                                                            {
                                                                "text": "Creator Channel",
                                                                "navigationEndpoint": {
                                                                    "commandMetadata": {
                                                                        "webCommandMetadata": {
                                                                            "url": "/@creator"
                                                                        }
                                                                    }
                                                                },
                                                            }
                                                        ]
                                                    },
                                                    "viewCountText": {"simpleText": "1.5M views"},
                                                    "publishedTimeText": {"simpleText": "4 hours ago"},
                                                    "lengthText": {"simpleText": "10:15"},
                                                    "thumbnail": {
                                                        "thumbnails": [
                                                            {"url": "https://i.ytimg.com/vi/abc123XYZ00/hqdefault.jpg"}
                                                        ]
                                                    },
                                                }
                                            },
                                            {
                                                "shortsLockupViewModel": {
                                                    "entityId": "shorts-shelf-item-short999AA",
                                                    "overlayMetadata": {
                                                        "primaryText": {"content": "Viral Short"},
                                                        "secondaryText": {"content": "800K views"},
                                                    },
                                                }
                                            },
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
        html = f"<html><script>var ytInitialData = {json.dumps(sample_data)};</script></html>"
        extracted = yt.extract_yt_initial_data(html)
        self.assertIsNotNone(extracted)

        items = yt.collect_videos_from_yt_data(extracted, limit=10)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["video_id"], "abc123XYZ00")
        self.assertEqual(items[0]["title"], "Trending Video")
        self.assertEqual(items[0]["channel"], "Creator Channel")
        self.assertEqual(items[0]["view_count"], 1_500_000)
        self.assertFalse(items[0]["is_short"])

        self.assertEqual(items[1]["video_id"], "short999AA")
        self.assertTrue(items[1]["is_short"])

    def test_fetch_trending_caches_results(self):
        sample_html = (
            '<html><script>var ytInitialData = {"contents": [{"videoRenderer": '
            '{"videoId": "vid12345678", "title": {"simpleText": "Hot Topic"}, '
            '"lengthText": {"simpleText": "5:00"}, "viewCountText": {"simpleText": "50K views"}}}]};</script></html>'
        )
        with patch.object(yt, "_fetch_youtube_html", return_value=sample_html) as mock_fetch:
            res1 = yt.fetch_trending_or_search(region="US", category="all")
            res2 = yt.fetch_trending_or_search(region="US", category="all")

        self.assertEqual(mock_fetch.call_count, 1)
        self.assertFalse(res1["cached"])
        self.assertTrue(res2["cached"])
        self.assertEqual(len(res2["items"]), 1)
        self.assertEqual(res2["items"][0]["video_id"], "vid12345678")


if __name__ == "__main__":
    unittest.main()
