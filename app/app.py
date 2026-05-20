from __future__ import annotations

from app.ui.layout import build_app


def main() -> None:
    app = build_app()
    app.queue().launch()


if __name__ == "__main__":
    main()
