from __future__ import annotations

import argparse
import os

from app.core.model_paths import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_SOURCE,
    SUPPORTED_MODEL_SOURCES,
    resolve_model_dir,
    resolve_model_source,
)
from app.core.onnx_bundles import ensure_onnx_bundle
from app.models.downloads import iter_downloadable_onnx_bundle_specs
from app.models.registry import default_registry


def parse_download_models_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CCBTag 所需模型到本地缓存。")
    parser.add_argument(
        "--source",
        choices=list(SUPPORTED_MODEL_SOURCES),
        default=os.getenv("CCBTAG_MODEL_SOURCE", DEFAULT_MODEL_SOURCE),
        help="下载源。",
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("CCBTAG_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="自定义模型缓存目录，默认 ./model。",
    )
    return parser.parse_args()


def download_model_bundles(model_dir: str, source: str) -> None:
    resolved_source = resolve_model_source(source)
    resolved_dir = str(resolve_model_dir(model_dir))
    registry = default_registry()
    for config, spec in iter_downloadable_onnx_bundle_specs(registry.configs, resolved_source):
        print(f"下载 {config.display_name}: {resolved_source} -> {spec.repo_id}")
        ensure_onnx_bundle(spec, resolved_source, model_dir=resolved_dir)


def download_models_main() -> None:
    args = parse_download_models_args()
    download_model_bundles(args.model_dir, args.source)
    print("模型下载完成")
