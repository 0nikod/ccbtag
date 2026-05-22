from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.settings import AppConfig, load_app_config
from app.models.registry import ModelRegistry, default_registry
from app.services.batch_service import BatchService
from app.services.dataset_service import DatasetService
from app.services.generation_service import GenerationService
from app.services.save_service import SaveService
from app.services.tag_edit_service import TagEditService


@dataclass
class AppServices:
    config: AppConfig
    registry: ModelRegistry
    dataset: DatasetService
    generation: GenerationService
    batch: BatchService
    save: SaveService
    tag_edit: TagEditService


def create_app_services(
    config_root: str | Path | None = None,
    registry: ModelRegistry | None = None,
) -> AppServices:
    config = load_app_config(config_root)
    active_registry = registry or default_registry()
    dataset = DatasetService(config.caption.joiner)
    generation = GenerationService(config, active_registry, config.tag)
    return AppServices(
        config=config,
        registry=active_registry,
        dataset=dataset,
        generation=generation,
        batch=BatchService(generation),
        save=SaveService(config, dataset),
        tag_edit=TagEditService(config.tag, config.caption.joiner),
    )
