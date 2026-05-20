# CCBTag

CCBTag 是一个本地 Gradio 图片 caption 编辑器，用于辅助制作 Stable
Diffusion / LoRA 训练数据集。每张图片的 caption 在内部拆成两个部分：

```text
tags + natural language
```

导出的 `.txt` 固定为 tags 在前、自然语言描述在后：

```text
1girl, solo, long hair. A girl with long hair is sitting indoors.
```

## 快速启动

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra models --group dev
UV_CACHE_DIR=/tmp/uv-cache uv run python -m app.app
```

默认使用 Hugging Face 缓存。若要指定模型缓存目录：

```bash
export CCBTAG_MODEL_DIR=/path/to/model-cache
```

## 自然语言描述模型

ToriiGate 0.5 按外部图文服务接入，不在 Gradio 进程内直接加载 5B
模型。启动 vLLM、llama.cpp 或其他 OpenAI-compatible 服务后，配置：

```bash
export CCBTAG_NL_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export CCBTAG_NL_MODEL=Minthy/ToriiGate-0.5
export CCBTAG_NL_API_KEY=optional
```

如果没有配置服务，Tag 功能仍可使用，但生成 NL 时会显示可读错误。

## 下载模型

使用 Hugging Face：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/download_models.py --source hf
```

使用 ModelScope：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/download_models.py --source modelscope
```

也可以指定缓存目录：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/download_models.py --source hf --model-dir ./models
```

## 保存格式

对 `image.png`，默认保存：

```text
image.txt
caption_json/image.caption.json
```

`.txt` 用于训练；`.caption.json` 用于恢复 tags、NL、模型来源和编辑状态。
界面中可以选择把元数据保存到同目录：

```text
image.caption.json
```

## 当前 Git 环境说明

当前工作区存在一个只读 `.git` 占位目录，普通 `git status` 会失败。
本项目实际 Git 元数据初始化在 `.repo-git` 中。此环境下使用：

```bash
git --git-dir=.repo-git --work-tree=. status
```

如果后续移除了只读 `.git` 占位，可以把仓库迁回标准 Git 布局。
