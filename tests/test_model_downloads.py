from __future__ import annotations

from pathlib import Path

from app.models.base import ModelConfig
from app.models.downloads import (
    build_cl_tagger_bundle_spec,
    iter_downloadable_onnx_bundle_specs,
    resolve_download_repo_id,
)
from app.models.registry import ModelRegistry


def _cl_config() -> ModelConfig:
    return ModelConfig(
        id="cl_tagger_1_02",
        display_name="CL Tagger 1.02",
        task="tag",
        source="local",
        backend="onnx",
        model_path="cella110n/cl_tagger",
        entry="CLTaggerOnnx",
        extras={
            "model_name": "cl_tagger_1_02",
            "download_sources": {
                "hf": "cella110n/cl_tagger",
                "modelscope": "nikoovo/cl-tagger",
            },
        },
    )


def test_resolve_download_repo_id_prefers_source_mapping() -> None:
    config = _cl_config()

    assert resolve_download_repo_id(config, "hf") == "cella110n/cl_tagger"
    assert resolve_download_repo_id(config, "modelscope") == "nikoovo/cl-tagger"


def test_cl_tagger_modelscope_bundle_only_targets_102() -> None:
    spec = build_cl_tagger_bundle_spec(_cl_config(), "modelscope")
    files = spec.required_files + spec.optional_files

    assert spec.repo_id == "nikoovo/cl-tagger"
    assert spec.required_files == ("cl_tagger_1_02/model.onnx",)
    assert all("cl_tagger_1_01" not in name for name in files)
    assert all("cl_tagger_1_00" not in name for name in files)


def test_iter_downloadable_onnx_bundle_specs_skips_external_models() -> None:
    registry = ModelRegistry(Path("app/config/models.json"))

    bundles = iter_downloadable_onnx_bundle_specs(registry.configs, "modelscope")

    assert [config.id for config, _ in bundles] == ["pixai_tagger_v0_9", "cl_tagger_1_02"]
