from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if any(
        getattr(handler, "_ccbtag_handler", False) for handler in root_logger.handlers
    ):
        _set_level(root_logger, level)
        _configure_library_levels()
        return

    handler = logging.StreamHandler(stream=sys.__stderr__)
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
