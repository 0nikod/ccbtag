from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models.base import BaseTagger, ModelInferenceError, ModelLoadError, TagPrediction


class PixaiOnnxTagger(BaseTagger):
    """PixAI adapter backed by dghs-imgutils.

    dghs-imgutils owns the ONNX session details and Hugging Face model cache.
    This class keeps those backend details out of the UI and normalizes output
    to the shared TagPrediction shape.
    """

    def load(self) -> None:
        try:
            from imgutils.tagging.pixai import get_pixai_tags
        except Exception as exc:  # pragma: no cover - depends on optional ML deps
            raise ModelLoadError(
                "PixAI Tagger 需要安装模型依赖: uv sync --extra models"
            ) from exc
        self._get_pixai_tags = get_pixai_tags
        self.loaded = True

    def predict(self, image: str | Path, **kwargs: Any) -> list[TagPrediction]:
        path = self._ensure_image_path(image)
        threshold = float(kwargs.get("threshold", 0.35))
        model_name = str(self.config.extras.get("model_name", "v0.9"))
        try:
            raw_result = self._get_pixai_tags(
                str(path),
                model_name=model_name,
                thresholds=threshold,
                fmt="tag",
            )
        except Exception as exc:  # pragma: no cover - real inference boundary
            raise ModelInferenceError(f"PixAI 推理失败: {exc}") from exc
        if not isinstance(raw_result, dict):
            raise ModelInferenceError("PixAI 返回格式不是 tag-score 字典")
        return [
            TagPrediction(tag=str(tag), score=float(score), source=self.id)
            for tag, score in raw_result.items()
        ]
