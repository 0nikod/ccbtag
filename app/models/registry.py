from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from app.models.base import BaseModel, ModelConfig, ModelLoadError


ENTRYPOINTS: dict[str, str] = {
    "PixaiOnnxTagger": "app.models.taggers.pixai_onnx:PixaiOnnxTagger",
    "CLTaggerOnnx": "app.models.taggers.cl_tagger_onnx:CLTaggerOnnx",
    "ToriiGateHttpCaptioner": "app.models.captioners.toriigate_http:ToriiGateHttpCaptioner",
}


class ModelRegistry:
    """Lazy model registry used by Gradio event handlers.

    The UI talks in terms of model display names and ids. Backend details stay
    behind this registry so ONNX, PyTorch, or HTTP adapters can be swapped
    without changing the interface layer.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.configs = self._load_configs()
        self._instances: dict[str, BaseModel] = {}

    def list_taggers(self) -> list[ModelConfig]:
        return [config for config in self.configs if config.task == "tag"]

    def list_captioners(self) -> list[ModelConfig]:
        return [config for config in self.configs if config.task == "nl"]

    def display_choices(self, task: str) -> list[str]:
        return [config.display_name for config in self.configs if config.task == task]

    def config_by_display(self, task: str, display_name: str) -> ModelConfig:
        for config in self.configs:
            if config.task == task and config.display_name == display_name:
                return config
        raise KeyError(f"未知模型: {display_name}")

    def get_model(self, model_id: str) -> BaseModel:
        if model_id in self._instances:
            return self._instances[model_id]
        config = self._config_by_id(model_id)
        cls = self._load_entrypoint(config.entry)
        model = cls(config)
        model.load()
        self._instances[model_id] = model
        return model

    def _load_entrypoint(self, entry: str) -> type[BaseModel]:
        dotted = ENTRYPOINTS.get(entry)
        if dotted is None:
            raise ModelLoadError(f"未注册模型入口: {entry}")
        module_name, class_name = dotted.split(":", 1)
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise ModelLoadError(f"模型入口导入失败，请检查依赖安装: {entry}") from exc
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ModelLoadError(f"模型入口不存在: {entry}")
        return cls

    def get_by_display(self, task: str, display_name: str) -> BaseModel:
        config = self.config_by_display(task, display_name)
        return self.get_model(config.id)

    def _load_configs(self) -> list[ModelConfig]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data.get("models", [])
        if not isinstance(rows, list):
            raise ValueError("models.json 中的 models 必须是列表")
        return [ModelConfig.from_dict(row) for row in rows]

    def _config_by_id(self, model_id: str) -> ModelConfig:
        for config in self.configs:
            if config.id == model_id:
                return config
        raise KeyError(f"未知模型 id: {model_id}")


def default_registry() -> ModelRegistry:
    return ModelRegistry(Path(__file__).resolve().parents[1] / "config" / "models.json")
