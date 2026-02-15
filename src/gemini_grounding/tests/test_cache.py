import sys
import os
import unittest
import time
from unittest.mock import MagicMock, patch

# Add src to path if running directly
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from gemini_grounding.search import search, search_cache


class TestSearchCache(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        search_cache.clear()

        # Mock response data
        self.mock_response_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Search Result"}]},
                    "groundingMetadata": {
                        "groundingChunks": [],
                        "groundingSupports": [],
                    },
                }
            ]
        }

    @patch("gemini_grounding.search.session")
    def test_caching_behavior(self, mock_session):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response_data
        mock_response.status_code = 200
        mock_session.post.return_value = mock_response

        # First call (should hit API)
        result1 = search("test query", api_key="test_key")
        self.assertEqual(result1["text"], "Search Result")
        self.assertEqual(mock_session.post.call_count, 1)

        # Reset mock but keep return value
        mock_session.post.reset_mock()

        # Second call with same params (should hit cache)
        result2 = search("test query", api_key="test_key")
        self.assertEqual(result2["text"], "Search Result")
        self.assertEqual(mock_session.post.call_count, 0)

        # Third call with different query (should hit API)
        result3 = search("different query", api_key="test_key")
        self.assertEqual(mock_session.post.call_count, 1)

    @patch("gemini_grounding.search.session")
    def test_cache_key_excludes_retry(self, mock_session):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response_data
        mock_response.status_code = 200
        mock_session.post.return_value = mock_response

        # First call
        search("test query", api_key="test_key", retry_count=3)
        self.assertEqual(mock_session.post.call_count, 1)

        mock_session.post.reset_mock()

        # Second call with different retry_count (should still hit cache)
        search("test query", api_key="test_key", retry_count=5)
        self.assertEqual(mock_session.post.call_count, 0)


if __name__ == "__main__":
    unittest.main()
