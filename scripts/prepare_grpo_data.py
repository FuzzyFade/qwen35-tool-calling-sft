#!/usr/bin/env python3
"""
从 SFT 训练数据中提取 GRPO/GSPO 所需的 prompt + tools + expected 数据

输入: data/train.jsonl (SFT 格式, messages + tools)
输出: data/grpo_train.jsonl (GRPO 格式, prompt + tools + expected_tool + expected_args)

GRPO 只需要 prompt（模型自己生成回答），不需要标准答案的完整文本。
但需要 expected_tool 和 expected_args 来计算 reward。
"""

import json
import random
import hashlib
from pathlib import Path


def extract_grpo_samples(input_path: str, output_path: str, max_samples: int = 8000):
    """从 SFT 数据中提取 GRPO 训练样本"""

    samples = []
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                messages = row.get("messages", [])
                tools = row.get("tools")

                # GRPO 只用有工具调用的样本
                if not tools:
                    continue

                # 提取 system + user 作为 prompt
                prompt_messages = []
                expected_tool = None
                expected_args = {}

                for m in messages:
                    role = m.get("role")

                    if role == "system":
                        prompt_messages.append({
                            "role": "system",
                            "content": m.get("content", "")
                        })
                    elif role == "user" and not prompt_messages or \
                         (role == "user" and all(p["role"] != "user" for p in prompt_messages)):
                        # 只取第一条 user 消息作为 prompt
                        prompt_messages.append({
                            "role": "user",
                            "content": m.get("content", "")
                        })
                    elif role == "assistant" and m.get("tool_calls"):
                        # 提取期望的工具调用（用于 reward 计算）
                        tc = m["tool_calls"][0]
                        func = tc.get("function", tc)
                        expected_tool = func.get("name", "")
                        expected_args = func.get("arguments", {})
                        if isinstance(expected_args, str):
                            try:
                                expected_args = json.loads(expected_args)
                            except (json.JSONDecodeError, TypeError):
                                expected_args = {}
                        if not isinstance(expected_args, dict):
                            expected_args = {}
                        break  # 只取第一个 tool_call

                # 确保有 user prompt 和 expected tool
                if not prompt_messages or not expected_tool:
                    skipped += 1
                    continue

                # 确保至少有一条 user 消息
                has_user = any(m["role"] == "user" for m in prompt_messages)
                if not has_user:
                    skipped += 1
                    continue

                # 如果没有 system 消息，加一个
                if not any(m["role"] == "system" for m in prompt_messages):
                    prompt_messages.insert(0, {
                        "role": "system",
                        "content": "You are a helpful assistant with access to tools."
                    })

                sample = {
                    "prompt": prompt_messages,
                    "tools": tools,
                    "expected_tool": expected_tool,
                    "expected_args": expected_args,
                }

                samples.append(sample)

            except Exception:
                skipped += 1

    print(f"提取了 {len(samples)} 条 GRPO 样本 (跳过 {skipped} 条)")

    # 去重 (基于 user prompt 内容)
    seen = set()
    deduped = []
    for s in samples:
        user_content = next(
            (m["content"] for m in s["prompt"] if m["role"] == "user"), ""
        )
        h = hashlib.md5(user_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(s)

    print(f"去重后: {len(deduped)} 条")

    # 采样（GRPO 不需要太多数据，每个 prompt 要生成多个候选）
    random.seed(42)
    random.shuffle(deduped)
    final = deduped[:max_samples]

    print(f"最终采样: {len(final)} 条")

    # 写入
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已保存: {output_path}")

    # 统计
    tool_counts = {}
    for s in final:
        t = s["expected_tool"]
        tool_counts[t] = tool_counts.get(t, 0) + 1

    print(f"\n工具分布 (top 10):")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/train.jsonl")
    parser.add_argument("--output", default="./data/grpo_train.jsonl")
    parser.add_argument("--max_samples", type=int, default=8000)
    args = parser.parse_args()

    extract_grpo_samples(args.input, args.output, args.max_samples)
