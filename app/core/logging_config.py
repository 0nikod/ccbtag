from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


LOG_FORMAT = "%(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if any(
        getattr(handler, "_ccbtag_handler", False) for handler in root_logger.handlers
    ):
        _set_level(root_logger, level)
        _configure_library_levels()
        return

    handler = RichHandler(
        console=Console(file=sys.__stderr__),
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    handler._ccbtag_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(handler)
    _set_level(root_logger, level)
    _configure_library_levels()


def _set_level(logger: logging.Logger, level: int) -> None:
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)


def _configure_library_levels() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
