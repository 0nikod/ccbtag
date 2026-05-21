from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_DIR = Path("./model")


def resolve_model_dir(value: str | os.PathLike[str] | None = None) -> Path:
    configured = value or os.getenv("CCBTAG_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_MODEL_DIR.expanduser().resolve()


def apply_default_model_cache_env() -> None:
    model_dir = str(resolve_model_dir())
    os.environ.setdefault("HF_HOME", model_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", model_dir)
    os.environ.setdefault("MODELSCOPE_CACHE", model_dir)
