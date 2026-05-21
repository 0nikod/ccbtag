from __future__ import annotations

from collections.abc import Iterable

from app.core.onnx_bundles import OnnxBundleSpec
from app.models.base import ModelConfig, ModelLoadError


def resolve_download_repo_id(config: ModelConfig, source: str) -> str:
    mapping = config.extras.get("download_sources")
    if mapping is None:
        return config.model_path
    if not isinstance(mapping, dict):
        raise ModelLoadError(f"{config.display_name} 的 download_sources 必须是对象")
    repo_id = mapping.get(source) or config.model_path
    return str(repo_id)


def build_pixai_bundle_spec(config: ModelConfig, source: str) -> OnnxBundleSpec:
    return OnnxBundleSpec(
        repo_id=resolve_download_repo_id(config, source),
        required_files=("model.onnx", "selected_tags.csv", "preprocess.json"),
        optional_files=("thresholds.csv",),
    )


def build_cl_tagger_bundle_spec(config: ModelConfig, source: str) -> OnnxBundleSpec:
    model_name = str(config.extras.get("model_name", "cl_tagger_1_02"))
    return OnnxBundleSpec(
        repo_id=resolve_download_repo_id(config, source),
        required_files=(f"{model_name}/model.onnx",),
        optional_files=(
            "tag_mapping.json",
            "model.onnx",
            f"{model_name}/tag_mapping.json",
            f"{model_name}_tag_mapping.json",
            "selected_tags.json",
            f"{model_name}/selected_tags.json",
        ),
        local_dir_env="CCBTAG_CL_TAGGER_DIR",
    )


def build_onnx_bundle_spec(config: ModelConfig, source: str) -> OnnxBundleSpec:
    builders = {
        "PixaiOnnxTagger": build_pixai_bundle_spec,
        "CLTaggerOnnx": build_cl_tagger_bundle_spec,
    }
    builder = builders.get(config.entry)
    if builder is None:
        raise ModelLoadError(f"未注册可下载的 ONNX 模型入口: {config.entry}")
    return builder(config, source)


def iter_downloadable_onnx_bundle_specs(
    configs: Iterable[ModelConfig],
    source: str,
) -> list[tuple[ModelConfig, OnnxBundleSpec]]:
    bundles: list[tuple[ModelConfig, OnnxBundleSpec]] = []
    for config in configs:
        if config.source != "local" or config.backend != "onnx":
            continue
        bundles.append((config, build_onnx_bundle_spec(config, source)))
    return bundles
