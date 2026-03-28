import os
import unittest
import requests
from unittest.mock import MagicMock, patch

from gemini_grounding.search import search, ensure_initialized, _state


class TestSearchCache(unittest.TestCase):
    def setUp(self):
        ensure_initialized()
        # Clear cache before each test
        _state.search_cache.clear()
        self.env_patcher = patch.dict(
            os.environ, {"GEMINI_RETRY_UNTIL_SUCCESS": "false"}, clear=False
        )
        self.env_patcher.start()

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

    def tearDown(self):
        self.env_patcher.stop()

    @patch.object(_state, "session", create=True)
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

    @patch.object(_state, "session", create=True)
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

    @patch("gemini_grounding.search.time.sleep")
    @patch("gemini_grounding.search.random.uniform", return_value=0)
    @patch.object(_state, "session", create=True)
    def test_retry_until_success_retries_until_response(
        self, mock_session, mock_uniform, mock_sleep
    ):
        fail_exception = requests.exceptions.RequestException("temporary failure")

        success_response = MagicMock()
        success_response.json.return_value = self.mock_response_data
        success_response.status_code = 200

        mock_session.post.side_effect = [fail_exception, success_response]

        result = search(
            "test query",
            api_key="test_key",
            retry_count=0,
            retry_delay=0,
            retry_until_success=True,
        )

        self.assertEqual(result["text"], "Search Result")
        self.assertEqual(mock_session.post.call_count, 2)
        mock_sleep.assert_called_once_with(0)

    @patch("gemini_grounding.search.time.sleep")
    @patch("gemini_grounding.search.random.uniform", return_value=0)
    @patch.object(_state, "session", create=True)
    def test_cache_key_includes_retry_until_success(
        self, mock_session, mock_uniform, mock_sleep
    ):
        empty_response = MagicMock()
        empty_response.json.return_value = {"candidates": []}
        empty_response.status_code = 200

        success_response = MagicMock()
        success_response.json.return_value = self.mock_response_data
        success_response.status_code = 200

        mock_session.post.return_value = empty_response
        result1 = search("test query", api_key="test_key")

        self.assertEqual(result1["text"], "")
        self.assertEqual(mock_session.post.call_count, 1)

        mock_session.post.reset_mock()
        mock_session.post.side_effect = [success_response]

        result2 = search(
            "test query",
            api_key="test_key",
            retry_count=0,
            retry_delay=0,
            retry_until_success=True,
        )

        self.assertEqual(result2["text"], "Search Result")
        self.assertEqual(mock_session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
