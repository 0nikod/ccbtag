from __future__ import annotations

from app.ui.layout import APP_CSS, build_app


def main() -> None:
    app = build_app()
    app.queue().launch(css=APP_CSS)


if __name__ == "__main__":
    main()
