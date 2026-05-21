from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.cli import download_models_main, parse_download_models_args
from app.core.model_paths import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_SOURCE,
    apply_default_model_cache_env,
    resolve_model_dir,
    resolve_model_source,
)
from app.core.onnx_bundles import (
    OnnxBundleSpec,
    clear_onnx_session_cache,
    close_onnx_session,
    ensure_hf_onnx_bundle,
    open_onnx_session,
    resolve_bundle_dir,
    select_onnx_providers,
)
from scripts.download_models import parse_args


def test_default_model_dir_points_to_model_folder() -> None:
    assert DEFAULT_MODEL_DIR.as_posix() == "model"


def test_resolve_model_dir_defaults_to_repo_model_folder() -> None:
    assert resolve_model_dir() == Path.cwd() / "model"


def test_resolve_model_dir_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_DIR", "./custom-models")

    assert resolve_model_dir() == (Path.cwd() / "custom-models").resolve()


def test_parse_args_defaults_model_dir(monkeypatch) -> None:
    monkeypatch.delenv("CCBTAG_MODEL_DIR", raising=False)
    monkeypatch.setattr("sys.argv", ["download_models.py"])

    args = parse_args()

    assert args.model_dir == "model"


def test_parse_args_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_DIR", "./custom-models")
    monkeypatch.setattr("sys.argv", ["download_models.py"])

    args = parse_args()

    assert args.model_dir == "./custom-models"


def test_package_cli_parse_defaults_model_dir(monkeypatch) -> None:
    monkeypatch.delenv("CCBTAG_MODEL_DIR", raising=False)
    monkeypatch.setattr("sys.argv", ["ccbtag-download-models"])

    args = parse_download_models_args()

    assert args.model_dir == "model"
    assert args.source == DEFAULT_MODEL_SOURCE


def test_package_cli_parse_prefers_model_source_env(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "hf")
    monkeypatch.setattr("sys.argv", ["ccbtag-download-models"])

    args = parse_download_models_args()

    assert args.source == "hf"


def test_package_cli_explicit_source_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "hf")
    monkeypatch.setattr("sys.argv", ["ccbtag-download-models", "--source", "modelscope"])

    args = parse_download_models_args()

    assert args.source == "modelscope"


def test_resolve_model_source_defaults_to_modelscope(monkeypatch) -> None:
    monkeypatch.delenv("CCBTAG_MODEL_SOURCE", raising=False)

    assert resolve_model_source() == DEFAULT_MODEL_SOURCE


def test_resolve_model_source_normalizes_env(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "HF")

    assert resolve_model_source() == "hf"


def test_resolve_model_source_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("CCBTAG_MODEL_SOURCE", "s3")

    with pytest.raises(ValueError, match="不支持的模型下载源"):
        resolve_model_source()


def test_apply_default_model_cache_env_sets_all_defaults(monkeypatch) -> None:
    for key in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"]:
        monkeypatch.delenv(key, raising=False)

    apply_default_model_cache_env()

    expected = str(Path.cwd() / "model")
    assert os.environ["HF_HOME"] == expected
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == expected
    assert os.environ["MODELSCOPE_CACHE"] == expected


def test_apply_default_model_cache_env_respects_model_dir_override(monkeypatch) -> None:
    for key in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CCBTAG_MODEL_DIR", "./custom-models")

    apply_default_model_cache_env()

    expected = str((Path.cwd() / "custom-models").resolve())
    assert os.environ["HF_HOME"] == expected
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == expected
    assert os.environ["MODELSCOPE_CACHE"] == expected


def test_bundle_loader_skips_download_when_files_exist(tmp_path: Path) -> None:
    spec = OnnxBundleSpec(
        repo_id="owner/model",
        required_files=("model.onnx", "selected_tags.csv"),
        optional_files=("thresholds.csv",),
        subdir="v0.9",
    )
    root = tmp_path / "model" / "owner" / "model" / "v0.9"
    root.mkdir(parents=True)
    (root / "model.onnx").write_text("x", encoding="utf-8")
    (root / "selected_tags.csv").write_text("x", encoding="utf-8")
    (root / "thresholds.csv").write_text("x", encoding="utf-8")
    calls: list[str] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        calls.append(filename)
        return str(Path(local_dir) / filename)

    bundle = ensure_hf_onnx_bundle(spec, fake_download, model_dir=tmp_path / "model")

    assert calls == []
    assert bundle.require("model.onnx") == root / "model.onnx"
    assert bundle.get("thresholds.csv") == root / "thresholds.csv"


def test_bundle_loader_downloads_only_missing_files(tmp_path: Path) -> None:
    spec = OnnxBundleSpec(
        repo_id="owner/model",
        required_files=("model.onnx", "selected_tags.csv"),
        optional_files=("thresholds.csv",),
        subdir="v0.9",
    )
    root = tmp_path / "model" / "owner" / "model" / "v0.9"
    root.mkdir(parents=True)
    (root / "model.onnx").write_text("x", encoding="utf-8")
    calls: list[str] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        calls.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("downloaded", encoding="utf-8")
        return str(target)

    bundle = ensure_hf_onnx_bundle(spec, fake_download, model_dir=tmp_path / "model")

    assert calls == ["v0.9/selected_tags.csv", "v0.9/thresholds.csv"]
    assert bundle.require("selected_tags.csv").exists()


def test_cl_tagger_override_bypasses_shared_model_root(monkeypatch, tmp_path: Path) -> None:
    spec = OnnxBundleSpec(
        repo_id="cella110n/cl_tagger",
        required_files=("model.onnx",),
        subdir="cl_tagger_1_02",
        local_dir_env="CCBTAG_CL_TAGGER_DIR",
    )
    override = tmp_path / "override"
    monkeypatch.setenv("CCBTAG_CL_TAGGER_DIR", str(override))

    assert resolve_bundle_dir(spec) == override.resolve()


def test_select_onnx_providers_falls_back_to_cpu() -> None:
    assert select_onnx_providers(["AzureExecutionProvider"]) == ["CPUExecutionProvider"]


def test_open_onnx_session_reuses_cached_session(tmp_path: Path) -> None:
    clear_onnx_session_cache()
    model_path = tmp_path / "model.onnx"
    model_path.write_text("x", encoding="utf-8")
    calls: list[tuple[str, list[str]]] = []

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def fake_factory(path: str, *, providers: list[str]) -> object:
        calls.append((path, providers))
        return {"path": path, "providers": providers}

    first = open_onnx_session(model_path, FakeOrt, session_factory=fake_factory)
    second = open_onnx_session(model_path, FakeOrt, session_factory=fake_factory)

    assert first is second
    assert calls == [(str(model_path), ["CUDAExecutionProvider", "CPUExecutionProvider"])]


def test_close_onnx_session_removes_from_cache(tmp_path: Path) -> None:
    clear_onnx_session_cache()
    model_path = tmp_path / "model.onnx"
    model_path.write_text("x", encoding="utf-8")
    calls: list[str] = []

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    def fake_factory(path: str, *, providers: list[str]) -> object:
        calls.append(path)
        return {"path": path, "providers": providers}

    open_onnx_session(model_path, FakeOrt, session_factory=fake_factory)
    assert len(calls) == 1

    close_onnx_session(model_path)
    # The cache should be empty now for this path, so it creates a new one
    open_onnx_session(model_path, FakeOrt, session_factory=fake_factory)
    assert len(calls) == 2


def test_download_models_main_uses_default_modelscope(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.delenv("CCBTAG_MODEL_SOURCE", raising=False)
    monkeypatch.setattr("sys.argv", ["ccbtag-download-models"])
    monkeypatch.setattr("app.cli.download_model_bundles", lambda model_dir, source: calls.append((model_dir, source)))

    download_models_main()

    assert calls == [("model", DEFAULT_MODEL_SOURCE)]
