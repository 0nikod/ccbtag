from __future__ import annotations

from app.core.model_paths import apply_default_model_cache_env
from app.ui.layout import APP_CSS, SHORTCUT_JS, build_app


def main() -> None:
    apply_default_model_cache_env()
    app = build_app()
    app.queue().launch(css=APP_CSS, js=SHORTCUT_JS)


if __name__ == "__main__":
    main()
