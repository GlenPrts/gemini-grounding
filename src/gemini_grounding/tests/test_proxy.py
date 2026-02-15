import unittest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import resolve_url


class TestProxy(unittest.TestCase):
    def setUp(self):
        resolve_url.cache_clear()

    @patch("search.session")
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
            timeout=5,
            headers={"X-Proxy-Manual-Redirect": "true"},
        )
        self.assertEqual(result, "https://final-destination.com")

    @patch("search.session")
    @patch.dict(os.environ, {}, clear=True)
    def test_proxy_not_configured(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://direct-resolved.com"
        mock_session.head.return_value = mock_response

        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/bar"
        result = resolve_url(url)

        mock_session.head.assert_called_with(url, allow_redirects=True, timeout=5)
        self.assertEqual(result, "https://direct-resolved.com")


if __name__ == "__main__":
    unittest.main()
