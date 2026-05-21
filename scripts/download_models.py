from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import download_models_main as main
from app.cli import parse_download_models_args as parse_args


if __name__ == "__main__":
    main()
