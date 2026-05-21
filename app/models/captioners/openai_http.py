from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import openai

from app.models.base import BaseCaptioner, ModelInferenceError, ModelLoadError
from app.models.captioners.prompts import make_user_query, system_prompt


class OpenAIHttpCaptioner(BaseCaptioner):
    """Generic OpenAI-compatible HTTP captioner."""

    def load(self) -> None:
        endpoint_env = str(self.config.extras.get("endpoint_env", "OPENAI_BASE_URL"))
        model_env = str(self.config.extras.get("model_env", "OPENAI_MODEL"))
        api_key_env = str(self.config.extras.get("api_key_env", "OPENAI_API_KEY"))

        default_endpoint = str(
            self.config.extras.get(
                "default_endpoint",
                self.config.extras.get("ui_default_endpoint", "http://127.0.0.1:8000/v1/chat/completions"),
            )
        )
        default_model = str(self.config.extras.get("default_model_name", self.config.model_path))

        self.endpoint = os.getenv(endpoint_env, default_endpoint)
        self.model = os.getenv(model_env, default_model)
        self.api_key = os.getenv(api_key_env, "sk-dummy")
        self.timeout = float(os.getenv(str(self.config.extras.get("timeout_env", "OPENAI_TIMEOUT")), "120"))

        if not self.endpoint:
            raise ModelLoadError(f"未配置 NL 服务端点: {endpoint_env}")

        try:
            self.client = openai.OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key,
                timeout=self.timeout
            )
        except Exception as exc:
            raise ModelLoadError(f"OpenAI 客户端初始化失败: {exc}") from exc

        self.loaded = True

    def predict(self, image: str | Path, tags: list[str] | None = None, **kwargs: Any) -> str:
        path = self._ensure_image_path(image)
        user_prompt = self._prompt(tags or [], kwargs)

        endpoint = str(kwargs.get("endpoint") or self.endpoint)
        model = str(kwargs.get("model") or self.model)
        api_key = str(kwargs.get("api_key") or self.api_key)

        if endpoint != self.endpoint or api_key != self.api_key:
            client = openai.OpenAI(base_url=endpoint, api_key=api_key, timeout=self.timeout)
            client_used = client
        else:
            client_used = self.client

        try:
            response = client_used.chat.completions.create(
                model=model,
                messages=[
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
                max_tokens=int(kwargs.get("max_length", 300)),
                temperature=float(kwargs.get("temperature", 0.2)),
            )
            
            if not response.choices:
                raise ModelInferenceError("NL 服务返回缺少 choices")
                
            content = response.choices[0].message.content
            if content is None:
                raise ModelInferenceError("NL 服务返回缺少文本内容")
                
            return content.strip()
        except openai.OpenAIError as exc:
            raise ModelInferenceError(f"NL 服务调用失败: {exc}") from exc

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
            underscores_replace=underscores_replace,
            shuffle_tags=bool(kwargs.get("shuffle_tags", True)),
        )

    def _data_url(self, path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

