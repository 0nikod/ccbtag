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
uv sync --extra models --group dev
uv run ccbtag
```

PixAI 和 CL Tagger 由仓库内置的 ONNX loader 直接加载，默认模型目录为 `./model`。若要指定模型缓存目录：

```bash
export CCBTAG_MODEL_DIR=/path/to/model-cache
```

## 自然语言描述模型

默认配置的是通用的 OpenAI-compatible 接口，方便按外部图文服务接入。启动 vLLM、llama.cpp 或其他 OpenAI-compatible 服务后，配置：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1/chat/completions
export OPENAI_MODEL=gpt-3.5-turbo
export OPENAI_API_KEY=optional
```

如果没有配置服务，Tag 功能仍可使用，但生成 NL 时会显示可读错误。

## 下载模型

默认使用 ModelScope：

```bash
uv run ccbtag-download-models
```

显式使用 Hugging Face：

```bash
uv run ccbtag-download-models --source hf
```

也可以指定缓存目录：

```bash
uv run ccbtag-download-models --model-dir ./model
```

运行时缺少本地模型时也默认走 ModelScope。如需切换下载源：

```bash
export CCBTAG_MODEL_SOURCE=hf
```

`CL Tagger` 在 ModelScope 下只会下载 `nikoovo/cl-tagger` 仓库中的 `cl_tagger_1_02`。

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
