from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Raw model configuration from `app/config/models.json`."""

    id: str
    display_name: str
    task: str
    source: str
    backend: str
    model_path: str
    entry: str
    extras: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        required = [
            "id",
            "display_name",
            "task",
            "source",
            "backend",
            "model_path",
            "entry",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"模型配置缺少字段: {', '.join(missing)}")
        extras = {key: value for key, value in data.items() if key not in required}
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            task=str(data["task"]),
            source=str(data["source"]),
            backend=str(data["backend"]),
            model_path=str(data["model_path"]),
            entry=str(data["entry"]),
            extras=extras,
        )


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "name": self.tag,
            "score": self.score,
            "source": self.source,
        }


class ModelError(RuntimeError):
    """Base class for user-facing model failures."""


class ModelLoadError(ModelError):
    """Raised when dependencies, files, or remote models are unavailable."""


class ModelInferenceError(ModelError):
    """Raised when a loaded model cannot process the selected image."""


class BaseModel:
    task: str = ""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.loaded = False

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def display_name(self) -> str:
        return self.config.display_name

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def _ensure_image_path(self, image: str | Path) -> Path:
        path = Path(image).expanduser()
        if not path.exists():
            raise ModelInferenceError(f"图片不存在: {path}")
        return path


class BaseTagger(BaseModel):
    task = "tag"

    def predict(self, image: str | Path, **kwargs: Any) -> list[TagPrediction]:
        raise NotImplementedError


class BaseCaptioner(BaseModel):
    task = "nl"

    def predict(
        self, image: str | Path, tags: list[str] | None = None, **kwargs: Any
    ) -> str:
        raise NotImplementedError
