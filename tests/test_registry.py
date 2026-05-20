import unittest
from pathlib import Path

from app.models.registry import ModelRegistry


class RegistryTest(unittest.TestCase):
    def test_registry_lists_display_choices(self) -> None:
        registry = ModelRegistry(Path("app/config/models.json"))

        self.assertIn("PixAI Tagger v0.9", registry.display_choices("tag"))
        self.assertIn("ToriiGate 0.5", registry.display_choices("nl"))


if __name__ == "__main__":
    unittest.main()
