from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.model_paths import resolve_model_source
from app.core.onnx_bundles import close_onnx_session, ensure_onnx_bundle, open_onnx_session
from app.models.base import BaseTagger, ModelInferenceError, ModelLoadError, TagPrediction
from app.models.downloads import build_cl_tagger_bundle_spec


class CLTaggerOnnx(BaseTagger):
    """ONNX adapter for `cella110n/cl_tagger` `cl_tagger_1_02`.

    The repository exposes ONNX files plus tag mappings. The adapter performs a
    conservative WD-style square resize and uses flexible JSON parsing because
    community tagger repositories differ slightly in mapping shape.
    """

    image_size = 448

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on optional ML deps
            raise ModelLoadError(
                "CL Tagger 需要安装模型依赖: uv sync --extra models"
            ) from exc

        source = resolve_model_source()
        bundle = ensure_onnx_bundle(build_cl_tagger_bundle_spec(self.config, source), source)
        onnx_path = self._find_onnx_file(bundle)
        if onnx_path is None:
            raise ModelLoadError(f"未找到 CL Tagger ONNX 文件: {bundle.root}")
        mapping_path = self._find_mapping_file(bundle)
        if mapping_path is None:
            raise ModelLoadError(f"未找到 CL Tagger tag_mapping.json: {bundle.root}")

        self.tags = self._load_mapping(mapping_path)
        self.onnx_path = onnx_path
        self.session = open_onnx_session(self.onnx_path, ort)
        self.input_name = self.session.get_inputs()[0].name
        self.loaded = True

    def unload(self) -> None:
        if hasattr(self, "session"):
            self.session = None
        if hasattr(self, "onnx_path"):
            close_onnx_session(self.onnx_path)
        super().unload()

    def predict(self, image: str | Path, **kwargs: Any) -> list[TagPrediction]:
        path = self._ensure_image_path(image)
        try:
            batch = self._preprocess(path)
            outputs = self.session.run(None, {self.input_name: batch})
        except Exception as exc:  # pragma: no cover - real inference boundary
            raise ModelInferenceError(f"CL Tagger 推理失败: {exc}") from exc

        scores = np.asarray(outputs[0]).reshape(-1).astype("float32")
        scores = self._probabilities(scores)
        count = min(len(self.tags), scores.shape[0])
        return [
            TagPrediction(tag=self.tags[index], score=float(scores[index]), source=self.id)
            for index in range(count)
        ]

    def _find_onnx_file(self, bundle: Any) -> Path | None:
        model_name = str(self.config.extras.get("model_name", "cl_tagger_1_02"))
        for name in [f"{model_name}/model.onnx", "model.onnx"]:
            path = bundle.get(name)
            if path is not None and path.exists():
                return path
        return None

    def _find_mapping_file(self, bundle: Any) -> Path | None:
        model_name = str(self.config.extras.get("model_name", "cl_tagger_1_02"))
        for name in [
            "tag_mapping.json",
            f"{model_name}/tag_mapping.json",
            f"{model_name}_tag_mapping.json",
            "selected_tags.json",
            f"{model_name}/selected_tags.json",
        ]:
            path = bundle.get(name)
            if path is not None and path.exists():
                return path
        return None

    def _load_mapping(self, path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        tags = self._extract_tags(data)
        if not tags:
            raise ModelLoadError(f"tag mapping 为空: {path}")
        return tags

    def _extract_tags(self, data: Any) -> list[str]:
        if isinstance(data, list):
            return [str(item.get("name", item.get("tag", item))) if isinstance(item, dict) else str(item) for item in data]
        if isinstance(data, dict):
            for key in ["tags", "tag_mapping", "labels", "id2label"]:
                value = data.get(key)
                if value:
                    return self._extract_tags(value)
            if all(str(key).isdigit() for key in data.keys()):
                return self._extract_tags([data[str(index)] for index in range(len(data))])
            if all(str(value).isdigit() for value in data.values()):
                pairs = sorted(((int(index), tag) for tag, index in data.items()), key=lambda item: item[0])
                return [str(tag) for _, tag in pairs]
        return []

    def _preprocess(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            canvas = Image.new("RGB", (max(rgb.size), max(rgb.size)), (255, 255, 255))
            offset = ((canvas.width - rgb.width) // 2, (canvas.height - rgb.height) // 2)
            canvas.paste(rgb, offset)
            resized = canvas.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        array = np.asarray(resized).astype("float32") / 255.0
        return np.expand_dims(np.transpose(array, (2, 0, 1)), axis=0)

    def _probabilities(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
            return values
        clipped = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-clipped))
