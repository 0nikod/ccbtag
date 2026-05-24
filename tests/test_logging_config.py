from __future__ import annotations

import logging

from rich.logging import RichHandler

from app.core.logging_config import configure_logging


def test_configure_logging_installs_rich_handler() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        root_logger.handlers = []
        root_logger.setLevel(logging.NOTSET)

        configure_logging()

        assert any(isinstance(handler, RichHandler) for handler in root_logger.handlers)
        assert any(
            getattr(handler, "_ccbtag_handler", False)
            for handler in root_logger.handlers
        )
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
