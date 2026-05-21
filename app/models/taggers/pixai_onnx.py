from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.onnx_bundles import OnnxBundleSpec, ensure_hf_onnx_bundle, open_onnx_session
from app.models.base import BaseTagger, ModelInferenceError, ModelLoadError, TagPrediction


class PixaiOnnxTagger(BaseTagger):
    """Native PixAI ONNX adapter for the repository.

    The shared bundle loader only handles local-vs-download resolution plus the
    ONNX session lifecycle. PixAI-specific preprocessing and tag parsing stay
    here so the model semantics remain explicit.
    """

    def load(self) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - depends on optional ML deps
            raise ModelLoadError(
                "PixAI Tagger 需要安装模型依赖: uv sync --extra models"
            ) from exc

        bundle = ensure_hf_onnx_bundle(self._bundle_spec(), hf_hub_download)
        self.tags = self._load_tags(bundle.require("selected_tags.csv"))
        self.preprocess_steps = self._load_preprocess(bundle.require("preprocess.json"))
        self.bundle_thresholds = self._load_thresholds(bundle.get("thresholds.csv"))
        self.session = open_onnx_session(bundle.require("model.onnx"), ort)
        self.input_name = self.session.get_inputs()[0].name
        self.loaded = True

    def predict(self, image: str | Path, **kwargs: Any) -> list[TagPrediction]:
        path = self._ensure_image_path(image)
        try:
            batch = self._preprocess(path)
            outputs = self.session.run(None, {self.input_name: batch})
        except Exception as exc:  # pragma: no cover - real inference boundary
            raise ModelInferenceError(f"PixAI 推理失败: {exc}") from exc

        scores = self._pick_output(outputs, len(self.tags))
        scores = self._probabilities(scores)
        count = min(len(self.tags), scores.shape[0])
        return [
            TagPrediction(tag=self.tags[index]["name"], score=float(scores[index]), source=self.id)
            for index in range(count)
        ]

    def _bundle_spec(self) -> OnnxBundleSpec:
        return OnnxBundleSpec(
            repo_id=self.config.model_path,
            required_files=("model.onnx", "selected_tags.csv", "preprocess.json"),
            optional_files=("thresholds.csv",),
        )

    def _load_tags(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                rows.append(
                    {
                        "id": int(row.get("id", len(rows))),
                        "name": name,
                        "category": int(row.get("category", 0)),
                        "ips": self._parse_ips(row.get("ips", "")),
                    }
                )
        rows.sort(key=lambda item: int(item["id"]))
        if not rows:
            raise ModelLoadError(f"PixAI 标签文件为空: {path}")
        return rows

    def _load_preprocess(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            steps = data
        elif isinstance(data, dict):
            steps = data.get("stages") or data.get("pipeline") or data.get("preprocess") or data.get("steps")
        else:
            steps = None
        if not isinstance(steps, list) or not steps:
            raise ModelLoadError(f"PixAI preprocess.json 格式无效: {path}")
        return [self._normalize_step(step, path) for step in steps]

    def _load_thresholds(self, path: Path | None) -> dict[str, float]:
        if path is None or not path.exists():
            return {}
        thresholds: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = str(row.get("name") or row.get("category") or "").strip()
                value = row.get("threshold")
                if name and value is not None:
                    thresholds[name] = float(value)
        return thresholds

    def _normalize_step(self, step: Any, path: Path) -> dict[str, Any]:
        if not isinstance(step, dict):
            raise ModelLoadError(f"PixAI preprocess.json 中存在非法 stage: {path}")
        name = str(step.get("type") or step.get("name") or step.get("op") or "").strip().lower()
        if name not in {"resize", "to_tensor", "normalize"}:
            raise ModelLoadError(f"PixAI 暂不支持的预处理 stage: {name or '<empty>'}")
        return dict(step, type=name)

    def _parse_ips(self, raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw]
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []

    def _preprocess(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            value: Image.Image | np.ndarray = image.convert("RGB")
            for step in self.preprocess_steps:
                stage_type = str(step["type"])
                if stage_type == "resize":
                    value = self._apply_resize(value, step)
                elif stage_type == "to_tensor":
                    value = self._apply_to_tensor(value)
                elif stage_type == "normalize":
                    value = self._apply_normalize(value, step)
            if isinstance(value, Image.Image):
                value = self._apply_to_tensor(value)
        if value.ndim == 3:
            value = np.expand_dims(value, axis=0)
        return value.astype("float32", copy=False)

    def _apply_resize(self, value: Image.Image | np.ndarray, step: dict[str, Any]) -> Image.Image:
        image = value if isinstance(value, Image.Image) else Image.fromarray(self._to_hwc_uint8(value))
        size = step.get("size")
        if isinstance(size, int):
            target = (size, size)
        elif isinstance(size, list) and len(size) == 2:
            target = (int(size[0]), int(size[1]))
        elif isinstance(size, dict):
            target = (int(size.get("width", size.get("size", 448))), int(size.get("height", size.get("size", 448))))
        else:
            target = (
                int(step.get("width", step.get("resize", 448))),
                int(step.get("height", step.get("resize", 448))),
            )
        resample_name = str(step.get("resample") or step.get("interpolation") or "bicubic").upper()
        resample = getattr(Image.Resampling, resample_name, Image.Resampling.BICUBIC)
        return image.resize(target, resample)

    def _apply_to_tensor(self, value: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(value, Image.Image):
            array = np.asarray(value).astype("float32") / 255.0
        else:
            array = value.astype("float32", copy=False)
            max_value = float(array.max()) if array.size else 0.0
            min_value = float(array.min()) if array.size else 0.0
            if max_value > 1.0 or min_value < 0.0:
                array = array / 255.0
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                return array
        return np.transpose(array, (2, 0, 1))

    def _apply_normalize(self, value: Image.Image | np.ndarray, step: dict[str, Any]) -> np.ndarray:
        tensor = self._apply_to_tensor(value)
        mean = np.asarray(step.get("mean", [0.0, 0.0, 0.0]), dtype="float32").reshape(-1, 1, 1)
        std = np.asarray(step.get("std", [1.0, 1.0, 1.0]), dtype="float32").reshape(-1, 1, 1)
        return (tensor - mean) / std

    def _pick_output(self, outputs: list[Any], expected_dim: int) -> np.ndarray:
        candidates = [np.asarray(output) for output in outputs]
        for candidate in candidates:
            if candidate.ndim >= 2 and candidate.shape[-1] == expected_dim:
                return np.asarray(candidate[0]).reshape(-1).astype("float32")
            if candidate.ndim == 1 and candidate.shape[0] == expected_dim:
                return candidate.astype("float32")
        if not candidates:
            raise ModelInferenceError("PixAI 未返回任何输出")
        best = max(candidates, key=lambda item: item.shape[-1] if item.ndim else item.size)
        if best.ndim >= 2:
            best = best[0]
        return np.asarray(best).reshape(-1).astype("float32")

    def _probabilities(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
            return values
        clipped = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _to_hwc_uint8(self, value: np.ndarray) -> np.ndarray:
        array = value
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size else 0.0
            scaled = array * 255.0 if max_value <= 1.0 else array
            array = np.clip(scaled, 0.0, 255.0).astype("uint8")
        return array
