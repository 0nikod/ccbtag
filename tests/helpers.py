from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.core.settings import AppConfig, load_app_config
from app.services.batch_service import BatchService
from app.services.context import AppServices
from app.services.dataset_service import DatasetService
from app.services.generation_service import GenerationService
from app.services.save_service import SaveService
from app.services.tag_category_service import TagCategoryService
from app.services.tag_edit_service import TagEditService


def write_image(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


class PredictionStub:
    def __init__(self, tag: str, score: float, source: str = "test") -> None:
        self.tag = tag
        self.score = score
        self.source = source

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "name": self.tag,
            "score": self.score,
            "source": self.source,
        }


class FakeTagModel:
    def __init__(
        self,
        predictions: list[PredictionStub] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.predictions = predictions or [PredictionStub("solo", 0.9)]
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict(self, image_path: str, **kwargs: object) -> list[PredictionStub]:
        self.calls.append({"image_path": image_path, **kwargs})
        if self.error is not None:
            raise self.error
        return self.predictions


class FakeNlModel:
    def __init__(
        self, response: str = "generated nl", error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict(self, image_path: str, **kwargs: object) -> str:
        self.calls.append({"image_path": image_path, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


class FakeRegistry:
    def __init__(
        self,
        tag_model: FakeTagModel | None = None,
        nl_model: FakeNlModel | None = None,
        endpoint: str = "http://registry-endpoint/v1/chat/completions",
        model_name: str = "registry-model",
    ) -> None:
        self.tag_model = tag_model or FakeTagModel()
        self.nl_model = nl_model or FakeNlModel()
        self.endpoint = endpoint
        self.model_name = model_name

    def get_by_display(self, task: str, display_name: str) -> object:
        if task == "tag":
            return self.tag_model
        if task == "nl":
            return self.nl_model
        raise KeyError(task)

    def display_choices(self, task: str) -> list[str]:
        if task == "tag":
            return ["PixAI Tagger v0.9"]
        if task == "nl":
            return ["OpenAI Completions"]
        return []

    def list_captioners(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                extras={
                    "default_endpoint": self.endpoint,
                    "default_model_name": self.model_name,
                }
            )
        ]


def make_config(**overrides: object) -> AppConfig:
    config = load_app_config()
    for key, value in overrides.items():
        config = replace(config, **{key: value})
    return config


def build_services(
    config: AppConfig | None = None,
    registry: FakeRegistry | None = None,
    tag_categories: TagCategoryService | None = None,
) -> AppServices:
    active_config = config or load_app_config()
    active_registry = registry or FakeRegistry()
    dataset = DatasetService(active_config.caption.joiner)
    generation = GenerationService(
        active_config,
        active_registry,
        active_config.tag,
        tag_categories,
    )
    return AppServices(
        config=active_config,
        registry=active_registry,
        dataset=dataset,
        generation=generation,
        batch=BatchService(generation),
        save=SaveService(active_config, dataset),
        tag_edit=TagEditService(active_config.tag, active_config.caption.joiner),
    )
