from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.models.base import BaseCaptioner, ModelInferenceError, ModelLoadError
from app.models.captioners.prompts import make_user_query, system_prompt


class OpenAIHttpCaptioner(BaseCaptioner):
    """Generic OpenAI-compatible HTTP captioner."""

    def load(self) -> None:
        endpoint_env = str(self.config.extras.get("endpoint_env", "OPENAI_BASE_URL"))
        model_env = str(self.config.extras.get("model_env", "OPENAI_MODEL"))
        api_key_env = str(self.config.extras.get("api_key_env", "OPENAI_API_KEY"))

        self.endpoint = os.getenv(endpoint_env, "http://127.0.0.1:8000/v1/chat/completions")
        self.model = os.getenv(model_env, self.config.model_path)
        self.api_key = os.getenv(api_key_env, "")
        self.timeout = float(os.getenv(str(self.config.extras.get("timeout_env", "OPENAI_TIMEOUT")), "120"))

        if not self.endpoint:
            raise ModelLoadError(f"未配置 NL 服务端点: {endpoint_env}")
        self.loaded = True

    def predict(self, image: str | Path, tags: list[str] | None = None, **kwargs: Any) -> str:
        path = self._ensure_image_path(image)
        user_prompt = self._prompt(tags or [], kwargs)

        endpoint = str(kwargs.get("endpoint") or self.endpoint)
        model = str(kwargs.get("model") or self.model)
        api_key = str(kwargs.get("api_key") or self.api_key)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": self._data_url(path)}},
                    ],
                }
            ],
            "max_tokens": int(kwargs.get("max_length", 300)),
            "temperature": float(kwargs.get("temperature", 0.2)),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - external service boundary
            raise ModelInferenceError(f"NL 服务不可用: {exc}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - external service boundary
            raise ModelInferenceError("NL 服务返回的不是 JSON") from exc
        return self._extract_text(data)

    def _prompt(self, tags: list[str], kwargs: dict[str, Any]) -> str:
        c_type = str(kwargs.get("c_type", "short"))
        use_names = bool(kwargs.get("use_names", True))
        add_tags = bool(kwargs.get("use_tags_as_context", True))
        add_characters = bool(kwargs.get("add_characters", True))
        add_char_tags = bool(kwargs.get("add_char_tags", False))
        add_description = bool(kwargs.get("add_description", False))
        underscores_replace = bool(kwargs.get("underscores_replace", False))

        item = {
            "tags": tags,
            "characters": kwargs.get("characters", []),
            "char_p_tags": kwargs.get("char_p_tags", {"chars": {}, "skins": {}}),
            "char_descr": kwargs.get("char_descr", {"chars": {}, "skins": {}})
        }

        return make_user_query(
            item=item,
            c_type=c_type,
            use_names=use_names,
            add_tags=add_tags,
            add_characters=add_characters,
            add_char_tags=add_char_tags,
            add_description=add_description,
            underscores_replace=underscores_replace
        )

    def _data_url(self, path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelInferenceError("NL 服务返回缺少 choices")
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
        raise ModelInferenceError("NL 服务返回缺少文本内容")
