import json
import random
from pathlib import Path
from typing import Any, Mapping

CONFIG_PATH = Path(__file__).with_name("caption_prompts.json")
EMPTY_CHARACTER_TRAITS = {"chars": {}, "skins": {}}


def load_prompt_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load caption prompt templates and options from a JSON config file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ("system_prompt", "prompts", "prompts_names_only")
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Prompt config is missing required keys: {missing}")

    return config


_CONFIG = load_prompt_config()
prompts_b: dict[str, str] = _CONFIG["prompts"]
prompts_names_only: dict[str, bool] = _CONFIG["prompts_names_only"]
system_prompt: str = _CONFIG["system_prompt"]


def _as_trait_dict(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize character trait/description data to {'chars': {}, 'skins': {}}."""
    if value is None:
        return {"chars": {}, "skins": {}}

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {"chars": {}, "skins": {}}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Character trait fields must be dicts or valid JSON strings."
            ) from exc

    if not isinstance(value, Mapping):
        raise TypeError("Character trait fields must be mappings.")

    return {
        "chars": dict(value.get("chars", {})),
        "skins": dict(value.get("skins", {})),
    }


def _format_tags(tags: list[str], underscores_replace: bool) -> str:
    if underscores_replace:
        return ", ".join(tag.replace("_", " ") if len(tag) > 3 else tag for tag in tags)
    return " ".join(tags)


def _format_name(name: str, underscores_replace: bool) -> str:
    return name.replace("_", " ") if underscores_replace else name


def make_user_query(
    item: Mapping[str, Any],
    c_type: str,
    use_names: bool | None = None,
    add_tags: bool = True,
    add_characters: bool = True,
    add_char_tags: bool = False,
    add_description: bool = False,
    underscores_replace: bool = False,
    shuffle_tags: bool = True,
) -> str:
    """Build the user prompt for image captioning from metadata and JSON templates.

    Args:
        item: Image metadata. Expected keys include tags, characters, char_p_tags, char_descr.
        c_type: Prompt type key from caption_prompt_config.json -> prompts.
        use_names: Whether to use character names. If None, uses prompts_names_only[c_type].
        add_tags: Include booru tags in the prompt.
        add_characters: Include explicit character names from item['characters'].
        add_char_tags: Include known/popular character tags from item['char_p_tags'].
        add_description: Include character descriptions from item['char_descr'].
        underscores_replace: Replace underscores in long tags/names with spaces.
        shuffle_tags: Shuffle image tags before adding them to the prompt.
    """
    if c_type not in prompts_b:
        valid_types = ", ".join(sorted(prompts_b))
        raise ValueError(
            f"Unknown caption type: {c_type!r}. Valid types: {valid_types}"
        )

    if use_names is None:
        use_names = prompts_names_only.get(c_type, False)

    tags = list(item.get("tags", []))
    if shuffle_tags:
        random.shuffle(tags)
    tags_string = _format_tags(tags, underscores_replace)

    user_request = "# Captioning format:\n"
    user_request += prompts_b[c_type]
    user_request += "\n"

    if add_tags:
        user_request += f"# Booru tags for the image\n[{tags_string}]\n\n"

    if use_names:
        if add_characters:
            chars_tags = list(item.get("characters", []))
            chars_string = _format_tags(chars_tags, underscores_replace)

            user_request += (
                "# Characters on picture:\n"
                "Here are names/tags for characters from the picture, "
                f"make sure to use them: [{chars_string}].\n\n"
            )

            chars_popular_tags = _as_trait_dict(
                item.get("char_p_tags", EMPTY_CHARACTER_TRAITS)
            )
            chars_description = _as_trait_dict(
                item.get("char_descr", EMPTY_CHARACTER_TRAITS)
            )

            if chars_popular_tags["chars"] and (add_char_tags or add_description):
                user_request += "# Known traits for characters\n"

                if add_char_tags:
                    user_request += (
                        "Here are popular tags for each characters on picture:\n"
                    )
                    for c_name, c_tags in chars_popular_tags["chars"].items():
                        name = _format_name(c_name, underscores_replace)
                        tags_s = _format_tags(list(c_tags), underscores_replace)
                        user_request += f"{name}: [{tags_s}]\n"

                    if chars_popular_tags["skins"]:
                        user_request += "Extra tags for characters skins:\n"
                        for c_name, c_tags in chars_popular_tags["skins"].items():
                            name = _format_name(c_name, underscores_replace)
                            tags_s = _format_tags(list(c_tags), underscores_replace)
                            user_request += f"{name}: [{tags_s}]\n"

                elif add_description:
                    user_request += "Here are general descriptions for each characters on the picture:\n"
                    for c_name, c_descr in chars_description["chars"].items():
                        name = _format_name(c_name, underscores_replace)
                        user_request += f"## {name}\n{c_descr}\n\n"

                    if chars_description["skins"]:
                        user_request += "Here are also descriptions for specific skin of characters:\n"
                        for c_name, c_descr in chars_description["skins"].items():
                            name = _format_name(c_name, underscores_replace)
                            user_request += f"## {name}\n{c_descr}\n\n"
        else:
            user_request += (
                "# Characters on picture:\n"
                "Try to recognize the characters in the picture and use their names.\n"
            )

        user_request += "\n"
    else:
        user_request += (
            "# Characters on picture:\nAvoid to guess names for characters.\n"
        )

    return user_request
