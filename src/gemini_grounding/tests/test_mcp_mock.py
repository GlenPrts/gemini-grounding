import asyncio
import os
import unittest
from unittest.mock import patch, AsyncMock

from gemini_grounding.mcp_server import google_search


class TestMCP(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("gemini_grounding.mcp_server.search")
    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_uses_defaults(self, mock_init, mock_search):
        mock_search.return_value = {
            "text": "This is a test result [1].",
            "sources": [
                {"id": 1, "title": "Test Source", "url": "https://example.com"}
            ],
        }

        result = asyncio.run(google_search("test query"))

        mock_init.assert_called_once()
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
    @patch("gemini_grounding.mcp_server.search")
    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_reads_env_defaults(self, mock_init, mock_search):
        mock_search.return_value = {"text": "Result", "sources": []}

        asyncio.run(google_search("query"))

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
    @patch("gemini_grounding.mcp_server.search")
    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_explicit_params_override_env(self, mock_init, mock_search):
        mock_search.return_value = {"text": "Result", "sources": []}

        asyncio.run(
            google_search(
                "query",
                model="custom-model",
                retry_count=5,
                retry_delay=2.0,
                search_delay_min=1.0,
                search_delay_max=2.0,
                retry_until_success=True,
            )
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

    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_empty_query_raises_tool_error(self, mock_init):
        from mcp.server.fastmcp.exceptions import ToolError

        with self.assertRaises(ToolError):
            asyncio.run(google_search(""))

    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_too_long_query_raises_tool_error(self, mock_init):
        from mcp.server.fastmcp.exceptions import ToolError

        with self.assertRaises(ToolError):
            asyncio.run(google_search("x" * 2001))

    @patch("gemini_grounding.mcp_server.ensure_initialized")
    def test_google_search_invalid_delay_raises_tool_error(self, mock_init):
        from mcp.server.fastmcp.exceptions import ToolError

        with self.assertRaises(ToolError):
            asyncio.run(
                google_search("query", search_delay_min=5.0, search_delay_max=1.0)
            )


if __name__ == "__main__":
    unittest.main()
