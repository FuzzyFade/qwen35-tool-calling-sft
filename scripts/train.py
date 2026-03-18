#!/usr/bin/env python3
"""
Qwen3.5-9B-Base 工具调用 SFT 训练脚本 (NVIDIA GPU)
使用 Unsloth + TRL SFTTrainer

支持的 GPU:
  - RTX 3090/4090 (24GB) — bf16 LoRA
  - A100 40GB/80GB — bf16 LoRA / 全量微调
  - H100 — 最佳性能

用法:
  python train.py                          # 默认配置
  python train.py --max_steps 2000         # 自定义步数
  python train.py --per_device_train_batch_size 4  # 调整 batch size
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG = {
    # 模型
    "model_name": "Qwen/Qwen3.5-9B-Base",
    "max_seq_length": 4096,
    "load_in_16bit": True,

    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # DeltaNet (linear attention) 层的投影
        "in_proj_qkv", "in_proj_z", "out_proj",
    ],

    # 训练
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,  # 有效 batch size = 2 * 8 = 16
    "max_steps": 2000,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 100,
    "eval_strategy": "steps",
    "save_total_limit": 3,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "seed": 42,
    "dataloader_num_workers": 4,
    "optim": "adamw_8bit",

    # 数据
    "data_dir": "./data",
    "output_dir": "./output",
}


# ============================================================
# 数据处理
# ============================================================

def load_training_data(data_dir: str, tokenizer):
    """加载 JSONL 训练数据并转换为 tokenizer 的文本格式"""

    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据不存在: {train_path}")

    def load_and_format(filepath):
        """直接读 JSONL 并转为纯文本，绕过 datasets 的类型推断问题"""
        texts = []
        skipped = 0
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    messages = sample.get("messages", [])
                    tools = sample.get("tools") or None

                    cleaned = []
                    for m in messages:
                        msg = {"role": m["role"], "content": m.get("content")}
                        if m.get("tool_calls"):
                            tcs = []
                            for tc in m["tool_calls"]:
                                func = tc.get("function", tc)
                                args = func.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except (json.JSONDecodeError, TypeError):
                                        args = {}
                                if not isinstance(args, dict):
                                    args = {}
                                tcs.append({
                                    "type": "function",
                                    "function": {"name": func.get("name", ""), "arguments": args}
                                })
                            msg["tool_calls"] = tcs
                            msg["content"] = None
                        if m.get("name"):
                            msg["name"] = m["name"]
                        cleaned.append(msg)

                    text = tokenizer.apply_chat_template(
                        cleaned, tools=tools, tokenize=False, add_generation_prompt=False,
                    )
                    if text:
                        texts.append(text)
                except Exception:
                    skipped += 1

        print(f"  {filepath}: {len(texts)} OK, {skipped} skipped")
        return Dataset.from_dict({"text": texts})

    print("格式化训练数据...")
    train_dataset = load_and_format(train_path)

    valid_dataset = None
    if os.path.exists(valid_path):
        print("格式化验证数据...")
        valid_dataset = load_and_format(valid_path)

    print(f"格式化后训练集: {len(train_dataset)} 样本")
    if valid_dataset:
        print(f"格式化后验证集: {len(valid_dataset)} 样本")

    return train_dataset, valid_dataset


# ============================================================
# 主训练流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Tool-Calling SFT Training")

    # 模型参数
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"])

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"])

    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=DEFAULT_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--num_train_epochs", type=int, default=None,
                        help="如果设置，则覆盖 max_steps，跑完整 epoch")

    # 数据路径
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])

    # 可选
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="qwen35-tool-calling-sft")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 wandb 日志")

    args = parser.parse_args()

    # ================================================================
    # 1. 加载模型
    # ================================================================
    print("=" * 60)
    print("Qwen3.5-9B-Base 工具调用 SFT 训练")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print()

    print("[1/4] 加载模型...")
    load_16bit = DEFAULT_CONFIG["load_in_16bit"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not load_16bit,
        load_in_16bit=load_16bit,
        dtype=None,
    )

    # ================================================================
    # 2. 配置 LoRA
    # ================================================================
    print("[2/4] 配置 LoRA 适配器...")

    # Unsloth 会自动检测哪些 target_modules 存在于模型中
    # 对于 Qwen3.5 的混合架构，有些层有 q_proj (full attention)，
    # 有些层有 in_proj_qkv (DeltaNet)，Unsloth 会自动处理
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=DEFAULT_CONFIG["lora_dropout"],
        target_modules=DEFAULT_CONFIG["target_modules"],
        use_gradient_checkpointing="unsloth",  # Unsloth 优化的梯度检查点
        random_state=DEFAULT_CONFIG["seed"],
    )

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ================================================================
    # 3. 加载数据
    # ================================================================
    print("[3/4] 加载训练数据...")
    train_dataset, valid_dataset = load_training_data(args.data_dir, tokenizer)

    # ================================================================
    # 4. 配置训练器
    # ================================================================
    print("[4/4] 启动训练...")

    # wandb 配置
    report_to = "wandb" if not args.no_wandb else "none"
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps if args.num_train_epochs is None else -1,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else 1,
        learning_rate=args.learning_rate,
        warmup_ratio=DEFAULT_CONFIG["warmup_ratio"],
        lr_scheduler_type=DEFAULT_CONFIG["lr_scheduler_type"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
        fp16=DEFAULT_CONFIG["fp16"],
        bf16=DEFAULT_CONFIG["bf16"],
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        save_steps=DEFAULT_CONFIG["save_steps"],
        eval_steps=DEFAULT_CONFIG["eval_steps"] if valid_dataset else None,
        eval_strategy=DEFAULT_CONFIG["eval_strategy"] if valid_dataset else "no",
        save_total_limit=DEFAULT_CONFIG["save_total_limit"],
        gradient_checkpointing=DEFAULT_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs=DEFAULT_CONFIG["gradient_checkpointing_kwargs"],
        seed=DEFAULT_CONFIG["seed"],
        dataloader_num_workers=DEFAULT_CONFIG["dataloader_num_workers"],
        optim=DEFAULT_CONFIG["optim"],
        max_seq_length=args.max_seq_length,
        report_to=report_to,
        run_name=f"qwen35-9b-tool-sft-r{args.lora_r}",
        dataset_text_field="text",
        packing=False,  # 工具调用数据不适合 packing
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )

    # 开始训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ================================================================
    # 5. 保存模型
    # ================================================================
    print()
    print("训练完成! 保存模型...")

    # 保存 LoRA 适配器
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  LoRA 适配器已保存: {adapter_dir}")

    # 保存合并后的完整模型 (bf16)
    merged_dir = os.path.join(args.output_dir, "merged_bf16")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"  合并模型 (bf16) 已保存: {merged_dir}")

    print()
    print("=" * 60)
    print("全部完成!")
    print(f"  适配器: {adapter_dir}")
    print(f"  完整模型: {merged_dir}")
    print()
    print("导出 GGUF (用于 Ollama/LM Studio):")
    print(f"  python export_gguf.py --model_dir {merged_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
