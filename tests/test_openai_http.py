import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai

from app.models.base import ModelConfig, ModelInferenceError, ModelLoadError
from app.models.captioners.openai_http import OpenAIHttpCaptioner
from app.models.captioners.prompts import make_user_query


class TestOpenAIHttpCaptioner(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            id="test_openai",
            task="nl",
            display_name="Test OpenAI",
            source="none",
            backend="http",
            model_path="gpt-3.5-turbo",
            entry="app.models.captioners.openai_http.OpenAIHttpCaptioner",
            extras={
                "endpoint_env": "TEST_OPENAI_BASE_URL",
                "model_env": "TEST_OPENAI_MODEL",
                "api_key_env": "TEST_OPENAI_API_KEY",
            },
        )

    @patch("os.getenv")
    @patch("openai.OpenAI")
    def test_load_success(self, mock_openai: MagicMock, mock_getenv: MagicMock) -> None:
        mock_getenv.side_effect = lambda key, default="": {
            "TEST_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "TEST_OPENAI_MODEL": "test-model",
            "TEST_OPENAI_API_KEY": "sk-123",
        }.get(key, default)

        captioner = OpenAIHttpCaptioner(self.config)
        captioner.load()

        self.assertTrue(captioner.loaded)
        mock_openai.assert_called_once_with(
            base_url="http://127.0.0.1:8000/v1", api_key="sk-123", timeout=120.0
        )
        self.assertEqual(captioner.model, "test-model")

    @patch("os.getenv")
    def test_load_missing_endpoint(self, mock_getenv: MagicMock) -> None:
        mock_getenv.side_effect = lambda key, default="": {
            "TEST_OPENAI_BASE_URL": "",  # Missing endpoint
            "OPENAI_TIMEOUT": "120",
        }.get(key, default)

        captioner = OpenAIHttpCaptioner(self.config)
        with self.assertRaises(ModelLoadError):
            captioner.load()

    @patch("os.getenv")
    @patch("openai.OpenAI")
    def test_predict_success(
        self, mock_openai: MagicMock, mock_getenv: MagicMock
    ) -> None:
        mock_getenv.side_effect = lambda key, default="": {
            "TEST_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "TEST_OPENAI_MODEL": "test-model",
            "TEST_OPENAI_API_KEY": "sk-123",
        }.get(key, default)

        captioner = OpenAIHttpCaptioner(self.config)
        captioner.load()

        # Mock the API response
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A girl standing."
        mock_client.chat.completions.create.return_value = mock_response

        # Mock file operations for data URL creation
        with (
            patch.object(Path, "read_bytes", return_value=b"fake-image-data"),
            patch.object(Path, "exists", return_value=True),
        ):
            result = captioner.predict(Path("test.png"), tags=["1girl", "standing"])

        self.assertEqual(result, "A girl standing.")
        mock_client.chat.completions.create.assert_called_once()
        kwargs_called = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs_called["model"], "test-model")
        self.assertEqual(len(kwargs_called["messages"]), 2)
        self.assertEqual(kwargs_called["messages"][1]["role"], "user")
        self.assertTrue(isinstance(kwargs_called["messages"][1]["content"], list))

    @patch("os.getenv")
    @patch("openai.OpenAI")
    def test_predict_openai_error(
        self, mock_openai: MagicMock, mock_getenv: MagicMock
    ) -> None:
        mock_getenv.side_effect = lambda key, default="": {
            "TEST_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        }.get(key, default)

        captioner = OpenAIHttpCaptioner(self.config)
        captioner.load()

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = openai.OpenAIError(
            "API Rate Limit"
        )

        with (
            patch.object(Path, "read_bytes", return_value=b"fake"),
            patch.object(Path, "exists", return_value=True),
        ):
            with self.assertRaises(ModelInferenceError) as context:
                captioner.predict(Path("test.png"))

        self.assertIn("API Rate Limit", str(context.exception))

    def test_make_user_query_does_not_mutate_tags(self) -> None:
        tags = ["1girl", "solo", "long_hair"]

        make_user_query(
            {"tags": tags},
            "short",
            False,
            True,
            False,
            False,
            False,
            shuffle_tags=True,
        )

        self.assertEqual(tags, ["1girl", "solo", "long_hair"])


if __name__ == "__main__":
    unittest.main()
