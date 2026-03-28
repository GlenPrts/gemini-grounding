import unittest
from unittest.mock import MagicMock, patch
import os
import requests

from gemini_grounding.search import resolve_url, ensure_initialized, _state


class TestProxy(unittest.TestCase):
    def setUp(self):
        ensure_initialized()
        resolve_url.cache_clear()

    @patch.object(_state, "resolve_session", create=True)
    @patch.dict(os.environ, {"GEMINI_PROXY_URL": "https://my-proxy.com"}, clear=True)
    def test_proxy_configured(self, mock_session):
        mock_response = MagicMock()
        mock_response.headers = {"X-Final-Url": "https://final-destination.com"}
        mock_session.head.return_value = mock_response

        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/foo"
        result = resolve_url(url)

        mock_session.head.assert_called_with(
            "https://my-proxy.com/https://vertexaisearch.cloud.google.com/grounding-api-redirect/foo",
            allow_redirects=False,
            timeout=_state.resolve_timeout,
            headers={"X-Proxy-Manual-Redirect": "true"},
        )
        self.assertEqual(result, "https://final-destination.com")

    @patch.object(_state, "resolve_session", create=True)
    @patch.dict(os.environ, {}, clear=True)
    def test_proxy_not_configured(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://direct-resolved.com"
        mock_session.head.return_value = mock_response

        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/bar"
        result = resolve_url(url)

        mock_session.head.assert_called_with(
            url, allow_redirects=True, timeout=_state.resolve_timeout
        )
        self.assertEqual(result, "https://direct-resolved.com")

    @patch.object(_state, "resolve_retry_count", 0)
    @patch.object(_state, "resolve_session", create=True)
    @patch.dict(os.environ, {"GEMINI_PROXY_URL": "https://my-proxy.com"}, clear=True)
    def test_proxy_timeout_returns_original_url(self, mock_session):
        mock_session.head.side_effect = requests.ReadTimeout("proxy timeout")

        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/baz"
        result = resolve_url(url)

        self.assertEqual(result, url)
        self.assertEqual(mock_session.head.call_count, 1)
        first_call = mock_session.head.call_args_list[0]
        self.assertEqual(
            first_call.args[0],
            "https://my-proxy.com/https://vertexaisearch.cloud.google.com/grounding-api-redirect/baz",
        )
        self.assertEqual(first_call.kwargs["allow_redirects"], False)


if __name__ == "__main__":
    unittest.main()
