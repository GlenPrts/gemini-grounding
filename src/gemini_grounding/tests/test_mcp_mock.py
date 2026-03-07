import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock mcp.server.fastmcp before importing server
fast_mcp_mock = MagicMock()
sys.modules["mcp.server.fastmcp"] = fast_mcp_mock


class DummyFastMCP:
    def __init__(self, name):
        self.name = name

    def tool(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


fast_mcp_mock.FastMCP.side_effect = DummyFastMCP

from mcp_server import google_search


class TestMCP(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("mcp_server.search")
    def test_google_search_uses_defaults(self, mock_search):
        mock_search.return_value = {
            "text": "This is a test result [1].",
            "sources": [
                {"id": 1, "title": "Test Source", "url": "https://example.com"}
            ],
        }

        result = google_search("test query")

        mock_search.assert_called_with(
            "test query",
            model="gemini-2.5-flash",
            retry_count=3,
            retry_delay=5.0,
            search_delay_min=0.0,
            search_delay_max=0.0,
            retry_until_success=False,
        )

        expected_output = "This is a test result [1].\n\n## Sources\n1. [Test Source](https://example.com)\n"
        self.assertEqual(result, expected_output)

    @patch.dict(
        os.environ,
        {
            "GEMINI_MODEL": "env-model",
            "GEMINI_RETRY_COUNT": "9",
            "GEMINI_RETRY_DELAY": "1.5",
            "GEMINI_SEARCH_DELAY_MIN": "0.25",
            "GEMINI_SEARCH_DELAY_MAX": "0.75",
            "GEMINI_RETRY_UNTIL_SUCCESS": "true",
        },
        clear=False,
    )
    @patch("mcp_server.search")
    def test_google_search_reads_env_defaults(self, mock_search):
        mock_search.return_value = {"text": "Result", "sources": []}

        google_search("query")

        mock_search.assert_called_with(
            "query",
            model="env-model",
            retry_count=9,
            retry_delay=1.5,
            search_delay_min=0.25,
            search_delay_max=0.75,
            retry_until_success=True,
        )

    @patch.dict(
        os.environ,
        {
            "GEMINI_MODEL": "env-model",
            "GEMINI_RETRY_COUNT": "9",
            "GEMINI_RETRY_DELAY": "1.5",
            "GEMINI_SEARCH_DELAY_MIN": "0.25",
            "GEMINI_SEARCH_DELAY_MAX": "0.75",
            "GEMINI_RETRY_UNTIL_SUCCESS": "false",
        },
        clear=False,
    )
    @patch("mcp_server.search")
    def test_google_search_explicit_params_override_env(self, mock_search):
        mock_search.return_value = {"text": "Result", "sources": []}

        google_search(
            "query",
            model="custom-model",
            retry_count=5,
            retry_delay=2.0,
            search_delay_min=1.0,
            search_delay_max=2.0,
            retry_until_success=True,
        )

        mock_search.assert_called_with(
            "query",
            model="custom-model",
            retry_count=5,
            retry_delay=2.0,
            search_delay_min=1.0,
            search_delay_max=2.0,
            retry_until_success=True,
        )


if __name__ == "__main__":
    unittest.main()
