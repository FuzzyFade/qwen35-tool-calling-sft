# Qwen3.5-9B-Base Tool-Calling SFT + GSPO

基于 Qwen3.5-9B-Base 的工具调用/Agent 能力微调项目。

两阶段训练：
1. **SFT** — 监督微调，教模型学会工具调用格式和模式
2. **GSPO** — 强化学习 (Group Sequence Policy Optimization)，让模型通过自我探索优化工具选择准确性

使用 Unsloth + TRL 框架，100k 中英双语工具调用数据，支持 Colab 一键训练。

## 训练数据

从 7 个公开数据集统一转换，共 **100,366 条**去重样本（90k 训练 / 10k 验证）：

| 数据集 | 数量 | 语言 | 类型 |
|--------|------|------|------|
| [Deepexi/function-calling-small](https://huggingface.co/datasets/Deepexi/function-calling-small) | 24,608 | 中文 | 阿里云 API 函数调用 |
| [llamafactory/glaive_toolcall_zh](https://huggingface.co/datasets/llamafactory/glaive_toolcall_zh) | 1,000 | 中文 | 工具调用多轮对话 |
| [hiyouga/glaive-function-calling-v2-sharegpt](https://huggingface.co/datasets/hiyouga/glaive-function-calling-v2-sharegpt) | 100,561 | 英文 | 函数调用 (ShareGPT) |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 1,893 | 英文 | Hermes 工具调用 |
| [tryumanshow/ToolACE-Qwen-cleaned](https://huggingface.co/datasets/tryumanshow/ToolACE-Qwen-cleaned) | 10,547 | 英文 | ToolACE (Qwen 格式) |
| [nohurry/Opus-4.6-Reasoning-3000x-filtered](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered) | 2,308 | 英文 | Claude Opus 推理蒸馏 |
| [bellfire/openclaw-coder-dataset](https://huggingface.co/datasets/bellfire/openclaw-coder-dataset) | 7,203 | 英文 | Agent 编排 |

其中 62.5% 含结构化工具定义，37.5% 为纯文本对话/推理。

## 快速开始

### Colab（推荐）

1. 上传 `Qwen35_9B_Tool_Calling_SFT.ipynb` 到 [Google Colab](https://colab.research.google.com/)
2. 选择 GPU 运行时（T4 免费 / A100 Pro）
3. 依次运行 Cell

Notebook 会自动检测 GPU 并调整训练参数：

| GPU | 量化 | batch | seq_len | 预计时间 |
|-----|------|-------|---------|---------|
| T4 (16GB) | 4-bit QLoRA | 1 | 2048 | ~4-6h |
| L4 (24GB) | 4-bit QLoRA | 2 | 4096 | ~2-3h |
| A100 (40GB+) | bf16 LoRA | 4 | 4096 | ~1h |

### NVIDIA 服务器

```bash
# 安装依赖
pip install -r requirements-nvidia.txt

# 准备数据（从 HuggingFace 下载 7 个数据集并转换）
python scripts/prepare_data.py

# 一键训练
bash run.sh

# 或自定义参数
python scripts/train.py \
  --max_steps 2000 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5
```

## 训练配置

| 参数 | 默认值 |
|------|--------|
| 基础模型 | `Qwen/Qwen3.5-9B-Base` |
| 方法 | LoRA (r=32, alpha=64) |
| 有效 batch size | 16 |
| 最大序列长度 | 4096 |
| 训练步数 | 2000 |
| 优化器 | AdamW 8-bit |
| 学习率 | 2e-5 (cosine) |
| 梯度检查点 | Unsloth 优化版 |

LoRA 目标层：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## 训练完成后

```bash
# 评测工具调用能力（中英文测试用例）
python scripts/eval_tool_calling.py --model_dir ./output/merged_bf16

# 导出 GGUF（用于 Ollama / LM Studio）
python scripts/export_gguf.py --model_dir ./output/merged_bf16
```

## 项目结构

```
├── Qwen35_9B_Tool_Calling_SFT.ipynb   # 阶段 1: SFT Notebook
├── Qwen35_9B_Tool_Calling_GSPO.ipynb  # 阶段 2: GSPO Notebook
├── scripts/
│   ├── prepare_data.py                 # SFT 数据准备（7 数据集 → 统一 JSONL）
│   ├── prepare_grpo_data.py            # GRPO 数据准备（提取 prompt + expected）
│   ├── train.py                        # SFT 训练脚本 (Unsloth + TRL)
│   ├── eval_tool_calling.py            # 工具调用评测
│   └── export_gguf.py                  # GGUF 导出
├── data/                               # 训练数据（gitignore，需本地生成）
│   ├── train.jsonl                     # SFT 数据 (90k)
│   ├── valid.jsonl                     # SFT 验证集 (10k)
│   └── grpo_train.jsonl                # GRPO 数据 (8k)
├── requirements-nvidia.txt
└── run.sh
```

## 两阶段训练

### 阶段 1: SFT（监督微调）

用标注数据教模型学会工具调用格式。详见 `Qwen35_9B_Tool_Calling_SFT.ipynb`。

### 阶段 2: GSPO（强化学习）

在 SFT 模型基础上，用 GSPO 让模型通过自我探索优化工具选择。详见 `Qwen35_9B_Tool_Calling_GSPO.ipynb`。

GSPO (Group Sequence Policy Optimization) 是 Qwen 团队提出的 GRPO 改进版，将重要性采样从 token 级改到 sequence 级。

**Reward 函数**:
- `tool_selection_reward` (权重 2.0): 选对了工具 +1.0, 调错 -0.5
- `format_reward` (权重 1.0): 输出格式合规 +1.0
- `args_reward` (权重 1.5): 参数匹配度 0.0~1.0

## 数据格式

所有数据统一为 mlx-lm / TRL 兼容的 JSONL 格式，通过 `tokenizer.apply_chat_template()` 自动转为 Qwen3.5 原生工具调用格式：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
    ]},
    {"role": "tool", "content": "{\"temp\": 22}", "name": "get_weather"},
    {"role": "assistant", "content": "The weather in Tokyo is 22°C."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather info", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
  ]
}
```

## 模型架构说明

Qwen3.5-9B-Base 不是标准 Transformer，采用混合架构：
- 32 层中 8 层为标准全注意力，24 层为 Gated DeltaNet（线性注意力）
- 内置 ViT 视觉编码器（早期融合多模态）
- 原生 262k token 上下文窗口
- 控制 token（`<|im_start|>`, `<|im_end|>`）已预训练，适合 LoRA 微调

## License

训练代码：MIT

基础模型和数据集各有独立许可证，请参考各自来源。
