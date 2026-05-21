from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.model_paths import DEFAULT_MODEL_DIR, resolve_model_dir


HF_MODELS = [
    "deepghs/pixai-tagger-v0.9-onnx",
    "cella110n/cl_tagger",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CCBTag 所需模型到本地缓存。")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf", help="下载源。")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("CCBTAG_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="自定义模型缓存目录，默认 ./model。",
    )
    return parser.parse_args()


def download_hf(model_dir: str) -> None:
    from huggingface_hub import snapshot_download

    cache_dir = str(resolve_model_dir(model_dir))
    for repo_id in HF_MODELS:
        print(f"下载 Hugging Face 模型: {repo_id}")
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


def download_modelscope(model_dir: str) -> None:
    from modelscope import snapshot_download

    cache_dir = str(resolve_model_dir(model_dir))
    for repo_id in HF_MODELS:
        print(f"下载 ModelScope 模型: {repo_id}")
        snapshot_download(repo_id, cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    if args.source == "hf":
        download_hf(args.model_dir)
    else:
        download_modelscope(args.model_dir)
    print("模型下载完成")


if __name__ == "__main__":
    main()
