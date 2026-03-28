import unittest
from unittest.mock import patch, MagicMock
import sys
import io
import os
import time
import requests

from gemini_grounding.search import search, main, ensure_initialized, _state


class TestSearchComprehensive(unittest.TestCase):
    def setUp(self):
        self.held_stdout = io.StringIO()
        self.held_stderr = io.StringIO()
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        sys.stdout = self.held_stdout
        sys.stderr = self.held_stderr

        ensure_initialized()
        _state.search_cache.clear()

        self.env_patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key"})
        self.env_patcher.start()

    def tearDown(self):
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original
        self.env_patcher.stop()

    @patch.object(_state, "session")
    def test_default_model(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_session.post.return_value = mock_response

        with patch("sys.argv", ["search.py", "--query", "test query"]):
            try:
                main()
            except SystemExit:
                pass

            args, _ = mock_session.post.call_args
            self.assertIn("gemini-2.5-flash", args[0])

    @patch.object(_state, "session")
    def test_model_argument_override(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_session.post.return_value = mock_response

        with patch(
            "sys.argv",
            ["search.py", "--query", "test query", "--model", "custom-model"],
        ):
            try:
                main()
            except SystemExit:
                pass

            args, _ = mock_session.post.call_args
            self.assertIn("custom-model", args[0])

    @patch.object(_state, "session")
    def test_env_var_model(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_session.post.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_MODEL": "env-model"}):
            with patch("sys.argv", ["search.py", "--query", "test query"]):
                try:
                    main()
                except SystemExit:
                    pass

                args, _ = mock_session.post.call_args
                self.assertIn("env-model", args[0])

    @patch.object(_state, "session")
    def test_retry_mechanism_success_after_failure(self, mock_session):
        fail_exception = requests.exceptions.RequestException(
            "Server Error", response=MagicMock(status_code=500)
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"candidates": []}

        mock_session.post.side_effect = [
            fail_exception,
            fail_exception,
            success_response,
        ]

        with patch.dict(
            os.environ, {"GEMINI_RETRY_DELAY": "0.1", "GEMINI_RETRY_COUNT": "3"}
        ):
            with patch("sys.argv", ["search.py", "--query", "test query"]):
                try:
                    main()
                except SystemExit:
                    pass

                self.assertEqual(mock_session.post.call_count, 3)

    @patch.object(_state, "session")
    def test_retry_exhaustion(self, mock_session):
        fail_exception = requests.exceptions.RequestException(
            "Service Unavailable", response=MagicMock(status_code=503)
        )

        mock_session.post.side_effect = fail_exception

        with patch.dict(
            os.environ, {"GEMINI_RETRY_DELAY": "0.1", "GEMINI_RETRY_COUNT": "2"}
        ):
            with patch("sys.argv", ["search.py", "--query", "test query"]):
                with self.assertRaises(SystemExit) as cm:
                    main()

                self.assertEqual(cm.exception.code, 1)
                self.assertEqual(mock_session.post.call_count, 3)

    @patch("gemini_grounding.search.time.sleep")
    @patch("gemini_grounding.search.random.uniform")
    @patch.object(_state, "session")
    def test_random_delay(self, mock_session, mock_uniform, mock_sleep):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_session.post.return_value = mock_response

        mock_uniform.return_value = 1.5

        with patch.dict(
            os.environ,
            {"GEMINI_SEARCH_DELAY_MIN": "1.0", "GEMINI_SEARCH_DELAY_MAX": "2.0"},
        ):
            with patch("sys.argv", ["search.py", "--query", "test query"]):
                try:
                    main()
                except SystemExit:
                    pass

                mock_uniform.assert_called_with(1.0, 2.0)
                mock_sleep.assert_any_call(1.5)

    @patch.object(_state, "session")
    def test_default_tool_config(self, mock_session):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_session.post.return_value = mock_response

        with patch("sys.argv", ["search.py", "--query", "test query"]):
            try:
                main()
            except SystemExit:
                pass

            args, kwargs = mock_session.post.call_args
            payload = kwargs["json"]
            tools = payload.get("tools", [])

            self.assertTrue(any("googleSearch" in tool for tool in tools))


if __name__ == "__main__":
    unittest.main()
