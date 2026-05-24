from __future__ import annotations

import json
import logging
import time
from importlib import import_module
from pathlib import Path

from app.models.base import BaseModel, ModelConfig, ModelLoadError


logger = logging.getLogger(__name__)


ENTRYPOINTS: dict[str, str] = {
    "PixaiOnnxTagger": "app.models.taggers.pixai_onnx:PixaiOnnxTagger",
    "CLTaggerOnnx": "app.models.taggers.cl_tagger_onnx:CLTaggerOnnx",
    "OpenAIHttpCaptioner": "app.models.captioners.openai_http:OpenAIHttpCaptioner",
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
        config = self._config_by_id(model_id)
        task = config.task

        # Unload other loaded models for the same task
        self.unload_task(task, keep_model_id=model_id)

        if model_id in self._instances:
            logger.info(
                "Reusing loaded model: task=%s model_id=%s display_name=%s",
                task,
                model_id,
                config.display_name,
            )
            return self._instances[model_id]

        cls = self._load_entrypoint(config.entry)
        model = cls(config)
        started = time.perf_counter()
        logger.info(
            "Loading model: task=%s model_id=%s display_name=%s backend=%s",
            task,
            model_id,
            config.display_name,
            config.backend,
        )
        model.load()
        self._instances[model_id] = model
        logger.info(
            "Loaded model: task=%s model_id=%s display_name=%s elapsed=%.3fs",
            task,
            model_id,
            config.display_name,
            time.perf_counter() - started,
        )
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

    def unload_task(self, task: str, keep_model_id: str | None = None) -> None:
        for existing_id, instance in list(self._instances.items()):
            if instance.config.task != task or existing_id == keep_model_id:
                continue
            self.unload_model(existing_id, instance)

    def unload_model(self, model_id: str, instance: BaseModel) -> None:
        logger.info(
            "Unloading model in 10 seconds: task=%s model_id=%s display_name=%s",
            instance.config.task,
            model_id,
            instance.display_name,
        )
        time.sleep(10)
        logger.info(
            "Unloading model: task=%s model_id=%s display_name=%s",
            instance.config.task,
            model_id,
            instance.display_name,
        )
        instance.unload()
        del self._instances[model_id]

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
