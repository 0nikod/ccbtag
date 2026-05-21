import unittest
from pathlib import Path

from app.models.registry import ModelRegistry


class RegistryTest(unittest.TestCase):
    def test_registry_lists_display_choices(self) -> None:
        registry = ModelRegistry(Path("app/config/models.json"))

        self.assertIn("PixAI Tagger v0.9", registry.display_choices("tag"))
        self.assertIn("ToriiGate 0.5", registry.display_choices("nl"))

    def test_registry_unloads_previous_model_with_same_task(self) -> None:
        from app.models.base import BaseModel
        
        class FakeModel(BaseModel):
            def load(self) -> None:
                self.loaded = True

        registry = ModelRegistry(Path("app/config/models.json"))
        
        def fake_load_entrypoint(entry: str) -> type[BaseModel]:
            return FakeModel
            
        registry._load_entrypoint = fake_load_entrypoint

        model1 = registry.get_model("pixai_tagger_v0_9")
        self.assertTrue(model1.loaded)
        
        model2 = registry.get_model("cl_tagger_1_02")
        self.assertTrue(model2.loaded)
        self.assertFalse(model1.loaded)
        self.assertNotIn("pixai_tagger_v0_9", registry._instances)
        self.assertIn("cl_tagger_1_02", registry._instances)

    def test_app_code_no_longer_imports_imgutils(self) -> None:
        for path in Path("app").rglob("*.py"):
            self.assertNotIn("imgutils", path.read_text(encoding="utf-8"), path.as_posix())


if __name__ == "__main__":
    unittest.main()
