import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock mcp.server.fastmcp before importing server
fast_mcp_mock = MagicMock()
sys.modules["mcp.server.fastmcp"] = fast_mcp_mock


def tool_decorator(func=None, **kwargs):
    if func and callable(func):
        return func

    def wrapper(f):
        return f

    return wrapper


mcp_instance_mock = MagicMock()
mcp_instance_mock.tool.side_effect = tool_decorator
fast_mcp_mock.FastMCP.return_value = mcp_instance_mock


from mcp_server import google_search


class TestMCP(unittest.TestCase):
    @patch("mcp_server.search")
    def test_google_search(self, mock_search):
        # Setup mock return value
        mock_search.return_value = {
            "text": "This is a test result [1].",
            "sources": [
                {"id": 1, "title": "Test Source", "url": "https://example.com"}
            ],
        }

        # Call the tool
        result = google_search("test query")
        print(f"DEBUG: result = {result}")

        import mcp_server

        print(f"DEBUG: server.search is {mcp_server.search}")
        print(f"DEBUG: mock_search is {mock_search}")

        # Verify arguments passed to search
        mock_search.assert_called_with(
            "test query",
            model="gemini-2.5-flash",
            retry_count=3,
            retry_delay=5.0,
            search_delay_min=0.0,
            search_delay_max=0.0,
        )

        # Verify output format
        expected_output = "This is a test result [1].\n\n## Sources\n1. [Test Source](https://example.com)\n"
        self.assertEqual(result, expected_output)

    @patch("mcp_server.search")
    def test_google_search_custom_params(self, mock_search):
        mock_search.return_value = {"text": "Result", "sources": []}

        google_search(
            "query", retry_count=5, search_delay_min=1.0, search_delay_max=2.0
        )

        mock_search.assert_called_with(
            "query",
            model="gemini-2.5-flash",
            retry_count=5,
            retry_delay=5.0,
            search_delay_min=1.0,
            search_delay_max=2.0,
        )


if __name__ == "__main__":
    unittest.main()
