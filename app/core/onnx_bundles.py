from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from app.core.model_paths import resolve_model_dir
from app.models.base import ModelLoadError


DownloadFile = Callable[..., str]
SessionFactory = Callable[..., Any]

_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


@dataclass(frozen=True)
class OnnxBundleSpec:
    repo_id: str
    required_files: tuple[str, ...]
    optional_files: tuple[str, ...] = ()
    subdir: str | None = None
    local_dir_env: str | None = None

    def repo_file(self, filename: str) -> str:
        candidate = PurePosixPath(filename)
        if self.subdir and len(candidate.parts) == 1:
            return str(PurePosixPath(self.subdir) / candidate)
        return str(candidate)


@dataclass(frozen=True)
class ResolvedOnnxBundle:
    root: Path
    files: dict[str, Path]

    def require(self, filename: str) -> Path:
        path = self.files.get(filename)
        if path is None:
            raise ModelLoadError(f"模型文件缺失: {filename} ({self.root})")
        return path

    def get(self, filename: str) -> Path | None:
        return self.files.get(filename)


def resolve_bundle_dir(spec: OnnxBundleSpec, model_dir: str | os.PathLike[str] | None = None) -> Path:
    override = os.getenv(spec.local_dir_env) if spec.local_dir_env else None
    if override:
        return Path(override).expanduser().resolve()
    return resolve_model_dir(model_dir) / spec.repo_id


def ensure_hf_onnx_bundle(
    spec: OnnxBundleSpec,
    download_file: DownloadFile,
    model_dir: str | os.PathLike[str] | None = None,
) -> ResolvedOnnxBundle:
    root = resolve_bundle_dir(spec, model_dir)
    root.mkdir(parents=True, exist_ok=True)

    required = {name: root / spec.repo_file(name) for name in spec.required_files}
    optional = {name: root / spec.repo_file(name) for name in spec.optional_files}

    if not _is_local_override(spec):
        missing_required = [name for name, path in required.items() if not path.exists()]
        missing_optional = [name for name, path in optional.items() if not path.exists()]
        for name in missing_required:
            _download_bundle_file(spec, root, name, download_file)
        for name in missing_optional:
            try:
                _download_bundle_file(spec, root, name, download_file)
            except Exception:
                continue

    missing_after_download = [name for name, path in required.items() if not path.exists()]
    if missing_after_download:
        joined = ", ".join(missing_after_download)
        raise ModelLoadError(f"模型文件缺失: {joined} (repo={spec.repo_id}, root={root})")

    files = {name: path for name, path in required.items()}
    files.update({name: path for name, path in optional.items() if path.exists()})
    return ResolvedOnnxBundle(root=root, files=files)


def select_onnx_providers(available: list[str] | tuple[str, ...]) -> list[str]:
    providers = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in available]
    return providers or ["CPUExecutionProvider"]


def open_onnx_session(
    onnx_path: Path,
    ort_module: Any,
    session_factory: SessionFactory | None = None,
) -> Any:
    providers = tuple(select_onnx_providers(list(ort_module.get_available_providers())))
    key = (str(onnx_path.expanduser().resolve()), providers)
    if key not in _SESSION_CACHE:
        factory = session_factory or ort_module.InferenceSession
        _SESSION_CACHE[key] = factory(str(onnx_path), providers=list(providers))
    return _SESSION_CACHE[key]


def clear_onnx_session_cache() -> None:
    _SESSION_CACHE.clear()


def _is_local_override(spec: OnnxBundleSpec) -> bool:
    return bool(spec.local_dir_env and os.getenv(spec.local_dir_env))


def _download_bundle_file(
    spec: OnnxBundleSpec,
    root: Path,
    filename: str,
    download_file: DownloadFile,
) -> None:
    download_file(
        repo_id=spec.repo_id,
        filename=spec.repo_file(filename),
        local_dir=str(root),
    )
