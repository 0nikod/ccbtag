from app.core.settings import load_app_config


def test_load_app_config_exposes_ui_defaults() -> None:
    config = load_app_config()

    assert config.caption.metadata_location == "caption_json"
    assert config.caption.joiner == ". "
    assert config.ui.nl_endpoint == "http://127.0.0.1:8000/v1/chat/completions"
    assert config.ui.nl_model_name == "gpt-3.5-turbo"
    assert config.ui.shuffle_tags is True
