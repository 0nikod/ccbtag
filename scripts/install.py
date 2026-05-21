from __future__ import annotations

import os
import subprocess
import sys


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, env=env)


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("请使用 Python 3.12 运行安装脚本，例如: uv run python scripts/install.py")

    env = os.environ.copy()
    run(["uv", "sync", "--extra", "models", "--group", "dev"], env)
    print()
    print("安装完成。启动应用:")
    print("uv run ccbtag")
    print("默认模型目录: ./model")
    print()
    print("如果使用本地 vLLM/llama.cpp 等 OpenAI 兼容服务，请设置:")
    print("export OPENAI_BASE_URL=http://127.0.0.1:8000/v1/chat/completions")
    print("export OPENAI_MODEL=gpt-3.5-turbo")


if __name__ == "__main__":
    main()
