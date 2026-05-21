from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_DIR = Path("./model")
DEFAULT_MODEL_SOURCE = "modelscope"
SUPPORTED_MODEL_SOURCES = ("hf", "modelscope")


def resolve_model_dir(value: str | os.PathLike[str] | None = None) -> Path:
    configured = value or os.getenv("CCBTAG_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_MODEL_DIR.expanduser().resolve()


def resolve_model_source(value: str | None = None) -> str:
    configured = value if value is not None else os.getenv("CCBTAG_MODEL_SOURCE", DEFAULT_MODEL_SOURCE)
    source = str(configured).strip().lower()
    if source not in SUPPORTED_MODEL_SOURCES:
        joined = ", ".join(SUPPORTED_MODEL_SOURCES)
        raise ValueError(f"不支持的模型下载源: {configured}，可选值: {joined}")
    return source


def apply_default_model_cache_env() -> None:
    model_dir = str(resolve_model_dir())
    os.environ.setdefault("HF_HOME", model_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", model_dir)
    os.environ.setdefault("MODELSCOPE_CACHE", model_dir)
