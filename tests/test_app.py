from __future__ import annotations

from app import app as app_module


class _QueuedApp:
    def __init__(self) -> None:
        self.launched_with: dict[str, object] | None = None

    def launch(self, **kwargs: object) -> None:
        self.launched_with = kwargs


class _FakeApp:
    def __init__(self) -> None:
        self.queued = _QueuedApp()
        self.queue_called = False

    def queue(self) -> _QueuedApp:
        self.queue_called = True
        return self.queued


def test_main_launches_with_css_and_js(monkeypatch) -> None:
    fake_app = _FakeApp()
    env_applied: list[bool] = []
    logging_configured: list[bool] = []

    monkeypatch.setattr(app_module, "build_app", lambda: fake_app)
    monkeypatch.setattr(
        app_module,
        "apply_default_model_cache_env",
        lambda: env_applied.append(True),
    )
    monkeypatch.setattr(
        app_module,
        "configure_logging",
        lambda: logging_configured.append(True),
    )

    app_module.main()

    assert logging_configured == [True]
    assert env_applied == [True]
    assert fake_app.queue_called is True
    assert fake_app.queued.launched_with == {
        "css": app_module.APP_CSS,
        "js": app_module.SHORTCUT_JS,
    }
