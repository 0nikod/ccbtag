import unittest
from pathlib import Path

from app.models.registry import ModelRegistry


class RegistryTest(unittest.TestCase):
    def test_registry_lists_display_choices(self) -> None:
        registry = ModelRegistry(Path("app/config/models.json"))

        self.assertIn("PixAI Tagger v0.9", registry.display_choices("tag"))
        self.assertIn("ToriiGate 0.5", registry.display_choices("nl"))

    def test_app_code_no_longer_imports_imgutils(self) -> None:
        for path in Path("app").rglob("*.py"):
            self.assertNotIn("imgutils", path.read_text(encoding="utf-8"), path.as_posix())


if __name__ == "__main__":
    unittest.main()
