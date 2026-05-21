from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.models.base import ModelConfig
from app.models.taggers.cl_tagger_onnx import CLTaggerOnnx


def _cl_config() -> ModelConfig:
    return ModelConfig(
        id="cl_tagger_1_02",
        display_name="CL Tagger 1.02",
        task="tag",
        source="local",
        backend="onnx",
        model_path="cella110n/cl_tagger",
        entry="CLTaggerOnnx",
        extras={"model_name": "cl_tagger_1_02"},
    )


def _install_fake_modules(monkeypatch, outputs: list[np.ndarray]) -> None:
    class FakeSession:
        def __init__(self, path: str, *, providers: list[str]) -> None:
            self.path = path
            self.providers = providers

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input")]

        def run(self, _outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert "input" in feed
            return outputs

    def fake_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        return str(Path(local_dir) / filename)

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=FakeSession,
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))


def _write_nested_bundle(root: Path) -> None:
    bundle = root / "cella110n" / "cl_tagger" / "cl_tagger_1_02"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_text("onnx", encoding="utf-8")
    (bundle / "tag_mapping.json").write_text(json.dumps(["1girl", "solo"]), encoding="utf-8")


def _write_flat_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.onnx").write_text("onnx", encoding="utf-8")
    (root / "tag_mapping.json").write_text(json.dumps(["1girl", "solo"]), encoding="utf-8")


def test_cl_tagger_predict_loads_standard_nested_bundle(tmp_path: Path, monkeypatch) -> None:
    _write_nested_bundle(tmp_path)
    _install_fake_modules(monkeypatch, [np.asarray([[2.0, -2.0]], dtype="float32")])
    monkeypatch.setenv("CCBTAG_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "hf")
    monkeypatch.delenv("CCBTAG_CL_TAGGER_DIR", raising=False)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)

    model = CLTaggerOnnx(_cl_config())
    model.load()
    predictions = model.predict(image_path)

    assert [item.tag for item in predictions] == ["1girl", "solo"]
    assert predictions[0].score == pytest.approx(0.8808, rel=1e-3)
    assert predictions[1].score == pytest.approx(0.1192, rel=1e-3)


def test_cl_tagger_local_override_accepts_flat_bundle_layout(tmp_path: Path, monkeypatch) -> None:
    _write_flat_bundle(tmp_path)
    _install_fake_modules(monkeypatch, [np.asarray([[2.0, -2.0]], dtype="float32")])
    monkeypatch.setenv("CCBTAG_CL_TAGGER_DIR", str(tmp_path))
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "hf")
    monkeypatch.delenv("CCBTAG_MODEL_DIR", raising=False)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)

    model = CLTaggerOnnx(_cl_config())
    model.load()
    predictions = model.predict(image_path)

    assert [item.tag for item in predictions] == ["1girl", "solo"]
