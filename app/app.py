from __future__ import annotations

import logging

from app.core.logging_config import configure_logging
from app.core.model_paths import apply_default_model_cache_env

configure_logging()

logger = logging.getLogger(__name__)
APP_CSS = ""
SHORTCUT_JS = ""


def build_app():
    _ensure_layout_exports()

    from app.ui.layout import build_app as layout_build_app

    return layout_build_app()


def _ensure_layout_exports() -> None:
    global APP_CSS, SHORTCUT_JS
    if APP_CSS and SHORTCUT_JS:
        return

    from app.ui.layout import APP_CSS as layout_css, SHORTCUT_JS as layout_js

    APP_CSS = layout_css
    SHORTCUT_JS = layout_js


def main() -> None:
    logger.info("Starting CCBTag GUI")
    apply_default_model_cache_env()
    _ensure_layout_exports()
    app = build_app()
    logger.info("Launching Gradio interface")
    app.queue().launch(css=APP_CSS, js=SHORTCUT_JS)


if __name__ == "__main__":
    main()
