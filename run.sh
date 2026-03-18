#!/usr/bin/env bash
# =============================================================
# Qwen3.5-9B-Base 工具调用 SFT 微调 (NVIDIA GPU)
# Unsloth + TRL SFTTrainer
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/output"

echo "============================================================"
echo "Qwen3.5-9B-Base Tool-Calling SFT (NVIDIA GPU)"
echo "============================================================"

# 检查 NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi 未找到。需要 NVIDIA GPU 和驱动。"
    exit 1
fi

echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 检查训练数据
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "训练数据不存在，运行数据准备脚本..."
    python "${SCRIPT_DIR}/scripts/prepare_data.py"
fi

TRAIN_LINES=$(wc -l < "${DATA_DIR}/train.jsonl")
echo "训练数据: ${TRAIN_LINES} 样本"
echo ""

# =============================================================
# 安装依赖 (首次运行)
# =============================================================
if ! python -c "import unsloth" 2>/dev/null; then
    echo "安装依赖..."
    pip install -r "${SCRIPT_DIR}/requirements-nvidia.txt"
fi

# =============================================================
# 开始训练
# =============================================================
echo "============================================================"
echo "开始 LoRA 训练..."
echo "============================================================"

python "${SCRIPT_DIR}/scripts/train.py" \
    --model_name "Qwen/Qwen3.5-9B-Base" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_seq_length 4096 \
    --lora_r 32 \
    --lora_alpha 64 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_steps 2000 \
    --learning_rate 2e-5 \
    "$@"

echo ""
echo "============================================================"
echo "训练完成!"
echo ""
echo "模型位置:"
echo "  LoRA 适配器:    ${OUTPUT_DIR}/final_adapter"
echo "  合并模型 (bf16): ${OUTPUT_DIR}/merged_bf16"
echo ""
echo "后续步骤:"
echo "  # 测试工具调用能力"
echo "  python scripts/eval_tool_calling.py --model_dir ${OUTPUT_DIR}/merged_bf16"
echo ""
echo "  # 导出 GGUF (用于 Ollama)"
echo "  python scripts/export_gguf.py --model_dir ${OUTPUT_DIR}/merged_bf16"
echo "============================================================"
