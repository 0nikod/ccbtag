from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.models.base import ModelConfig, ModelLoadError
from app.models.taggers.pixai_onnx import PixaiOnnxTagger


def _pixai_config() -> ModelConfig:
    return ModelConfig(
        id="pixai_tagger_v0_9",
        display_name="PixAI Tagger v0.9",
        task="tag",
        source="local",
        backend="onnx",
        model_path="deepghs/pixai-tagger-v0.9-onnx",
        entry="PixaiOnnxTagger",
        extras={"model_name": "v0.9"},
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


def _write_bundle(root: Path) -> None:
    bundle = root / "deepghs" / "pixai-tagger-v0.9-onnx"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_text("onnx", encoding="utf-8")
    (bundle / "selected_tags.csv").write_text(
        "id,name,category,ips\n0,1girl,0,[]\n1,hu_tao_(genshin_impact),4,[\"genshin_impact\"]\n",
        encoding="utf-8",
    )
    (bundle / "preprocess.json").write_text(
        '[{"type":"resize","size":[2,2]},{"type":"to_tensor"},{"type":"normalize","mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}]',
        encoding="utf-8",
    )
    (bundle / "thresholds.csv").write_text("category,threshold\ngeneral,0.3\ncharacter,0.85\n", encoding="utf-8")


def test_pixai_predict_returns_tag_predictions_from_local_bundle(tmp_path: Path, monkeypatch) -> None:
    _write_bundle(tmp_path)
    _install_fake_modules(monkeypatch, [np.asarray([[2.0, -2.0]], dtype="float32")])
    monkeypatch.setenv("CCBTAG_MODEL_DIR", str(tmp_path))
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)

    model = PixaiOnnxTagger(_pixai_config())
    model.load()
    predictions = model.predict(image_path, threshold=0.95)

    assert [item.tag for item in predictions] == ["1girl", "hu_tao_(genshin_impact)"]
    assert predictions[0].score == pytest.approx(0.8808, rel=1e-3)
    assert predictions[1].score == pytest.approx(0.1192, rel=1e-3)
    assert model.bundle_thresholds == {"general": 0.3, "character": 0.85}


def test_pixai_missing_required_file_has_readable_error(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "deepghs" / "pixai-tagger-v0.9-onnx"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_text("onnx", encoding="utf-8")
    (bundle / "selected_tags.csv").write_text("id,name,category,ips\n0,1girl,0,[]\n", encoding="utf-8")
    _install_fake_modules(monkeypatch, [np.asarray([[0.0]], dtype="float32")])
    monkeypatch.setenv("CCBTAG_MODEL_DIR", str(tmp_path))

    model = PixaiOnnxTagger(_pixai_config())

    with pytest.raises(ModelLoadError, match="preprocess.json"):
        model.load()


def test_pixai_unknown_preprocess_stage_fails_fast(tmp_path: Path, monkeypatch) -> None:
    _write_bundle(tmp_path)
    bundle = tmp_path / "deepghs" / "pixai-tagger-v0.9-onnx"
    (bundle / "preprocess.json").write_text('[{"type":"center_crop"}]', encoding="utf-8")
    _install_fake_modules(monkeypatch, [np.asarray([[0.0]], dtype="float32")])
    monkeypatch.setenv("CCBTAG_MODEL_DIR", str(tmp_path))

    model = PixaiOnnxTagger(_pixai_config())

    with pytest.raises(ModelLoadError, match="暂不支持的预处理 stage"):
        model.load()
