#!/usr/bin/env python3
"""
Qwen3.5-9B-Base 工具调用 SFT 数据准备脚本
统一将 7 个数据集转换为 mlx-lm 的 chat/tools JSONL 格式

目标格式 (每行一个 JSON):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"type": "function", "function": {"name": "...", "arguments": {...}}}
    ]},
    {"role": "tool", "content": "...", "name": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "...", "description": "...",
     "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
  ]
}

无 tools 的样本 (如推理蒸馏数据):
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import json
import hashlib
import random
import re
import sys
import os
from pathlib import Path
from collections import Counter

from datasets import load_dataset


# ============================================================
# 通用工具函数
# ============================================================

def wrap_tool_openai(tool_def: dict) -> dict:
    """将简单工具定义包装为 OpenAI 格式"""
    if "type" in tool_def and tool_def["type"] == "function" and "function" in tool_def:
        return tool_def  # 已经是 OpenAI 格式
    # 简单格式 → OpenAI 格式
    func = {}
    func["name"] = tool_def.get("name", "unknown")
    func["description"] = tool_def.get("description", "")
    params = tool_def.get("parameters", {})
    if isinstance(params, dict) and "type" in params and "properties" in params:
        func["parameters"] = params
    elif isinstance(params, dict):
        # 扁平参数格式 → OpenAI properties 格式
        properties = {}
        required = []
        for k, v in params.items():
            if isinstance(v, dict):
                properties[k] = v
                if v.get("required", False):
                    required.append(k)
                    v_copy = dict(v)
                    v_copy.pop("required", None)
                    properties[k] = v_copy
            else:
                properties[k] = {"type": "string", "description": str(v)}
        func["parameters"] = {"type": "object", "properties": properties}
        if required:
            func["parameters"]["required"] = required
    elif isinstance(params, list):
        # 列表格式参数
        properties = {}
        required = []
        for p in params:
            if isinstance(p, dict):
                name = p.get("name", p.get("parameter", "unknown"))
                ptype = p.get("type", "string")
                desc = p.get("description", "")
                properties[name] = {"type": ptype, "description": desc}
                if p.get("required", False):
                    required.append(name)
        func["parameters"] = {"type": "object", "properties": properties}
        if required:
            func["parameters"]["required"] = required
    else:
        func["parameters"] = {"type": "object", "properties": {}}
    return {"type": "function", "function": func}


def make_tool_call(name: str, arguments: dict) -> dict:
    """构建标准 tool_call 结构"""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    }


def msg_hash(messages: list) -> str:
    """计算 messages 内容的 hash 用于去重"""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def validate_sample(sample: dict) -> bool:
    """验证样本格式是否合法"""
    msgs = sample.get("messages", [])
    if not msgs or len(msgs) < 2:
        return False
    # 至少要有一条 user 和一条 assistant
    roles = [m.get("role") for m in msgs]
    if "user" not in roles:
        return False
    if "assistant" not in roles:
        return False
    # assistant 必须有 content 或 tool_calls
    for m in msgs:
        if m.get("role") == "assistant":
            has_content = m.get("content") is not None and m.get("content") != ""
            has_tools = bool(m.get("tool_calls"))
            if not has_content and not has_tools:
                return False
    return True


# ============================================================
# 转换器 1: Deepexi/function-calling-small (中文)
# ============================================================

def convert_deepexi(output_dir: Path) -> list:
    """转换 Deepexi 中文函数调用数据集"""
    print("[1/7] Loading Deepexi/function-calling-small ...")
    ds = load_dataset("Deepexi/function-calling-small", split="train")
    results = []

    for row in ds:
        try:
            system_prompt = row.get("systemPrompt", "")
            user_prompt = row.get("userPrompt", "")
            assistant_resp = row.get("assistantResponse", "")

            if not user_prompt or not assistant_resp:
                continue

            # 从 systemPrompt 提取工具定义
            # Deepexi 格式: 系统提示中嵌入了 JSON 工具定义
            tools = []
            # 尝试提取 JSON 块
            json_pattern = r'\{[^{}]*"function"[^{}]*"description"[^{}]*\}'
            found_tools = re.findall(json_pattern, system_prompt)
            for t in found_tools:
                try:
                    tool_def = json.loads(t)
                    tools.append(wrap_tool_openai(tool_def))
                except json.JSONDecodeError:
                    pass

            # 如果正则没提取到，把整个 system prompt 中的 JSON 数组找出来
            if not tools:
                array_pattern = r'\[(\{[^[\]]*\}(?:,\s*\{[^[\]]*\})*)\]'
                arrays = re.findall(array_pattern, system_prompt)
                for arr_str in arrays:
                    try:
                        arr = json.loads(f"[{arr_str}]")
                        for item in arr:
                            if isinstance(item, dict) and ("function" in item or "name" in item):
                                tools.append(wrap_tool_openai(item))
                    except json.JSONDecodeError:
                        pass

            # 解析 assistant 回复中的函数调用
            tool_calls = []
            try:
                resp = json.loads(assistant_resp)
                if isinstance(resp, dict):
                    func_name = resp.get("function", "unknown")
                    arguments = resp.get("arguments", {})
                    if isinstance(arguments, list) and len(arguments) > 0:
                        arguments = arguments[0] if isinstance(arguments[0], dict) else {"args": arguments}
                    elif isinstance(arguments, list):
                        arguments = {}
                    tool_calls.append(make_tool_call(func_name, arguments))
                elif isinstance(resp, list):
                    for r in resp:
                        if isinstance(r, dict) and "function" in r:
                            args = r.get("arguments", {})
                            if isinstance(args, list) and len(args) > 0:
                                args = args[0] if isinstance(args[0], dict) else {"args": args}
                            tool_calls.append(make_tool_call(r["function"], args))
            except json.JSONDecodeError:
                # 回复不是 JSON，作为普通文本处理
                pass

            messages = []
            # 用简洁的中文系统提示替代原始冗长的
            if tools:
                messages.append({"role": "system", "content": "你是一个有用的助手，可以调用工具来帮助用户完成任务。"})
            messages.append({"role": "user", "content": user_prompt})

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
            else:
                messages.append({"role": "assistant", "content": assistant_resp})

            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools

            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> Deepexi: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 2: llamafactory/glaive_toolcall_zh (中文)
# ============================================================

def convert_glaive_zh(output_dir: Path) -> list:
    """转换 llamafactory 中文工具调用数据集"""
    print("[2/7] Loading llamafactory/glaive_toolcall_zh ...")
    ds = load_dataset("llamafactory/glaive_toolcall_zh", split="train")
    results = []

    for row in ds:
        try:
            convs = row.get("conversations", [])
            tools_str = row.get("tools", "[]")

            # 解析 tools
            tools = []
            try:
                raw_tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw_tools, list):
                    for t in raw_tools:
                        tools.append(wrap_tool_openai(t))
            except (json.JSONDecodeError, TypeError):
                pass

            # 转换 conversations
            messages = []
            if tools:
                messages.append({"role": "system", "content": "你是一个有用的助手，可以调用工具来帮助用户完成任务。"})

            for conv in convs:
                role_from = conv.get("from", "")
                value = conv.get("value", "")

                if role_from == "human":
                    messages.append({"role": "user", "content": value})
                elif role_from == "gpt":
                    messages.append({"role": "assistant", "content": value})
                elif role_from == "function_call":
                    # 解析函数调用
                    try:
                        fc = json.loads(value) if isinstance(value, str) else value
                        tc = make_tool_call(fc.get("name", "unknown"), fc.get("arguments", {}))
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tc]
                        })
                    except (json.JSONDecodeError, TypeError):
                        messages.append({"role": "assistant", "content": value})
                elif role_from == "observation":
                    # 工具返回结果
                    tool_name = ""
                    # 尝试从上一条 assistant 消息获取工具名
                    for m in reversed(messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            tool_name = m["tool_calls"][0]["function"]["name"]
                            break
                    messages.append({
                        "role": "tool",
                        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                        "name": tool_name
                    })

            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools

            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> glaive_zh: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 3: hiyouga/glaive-function-calling-v2-sharegpt (英文)
# ============================================================

def convert_glaive_v2_sharegpt(output_dir: Path) -> list:
    """转换 glaive-function-calling-v2-sharegpt 英文工具调用数据集"""
    print("[3/7] Loading hiyouga/glaive-function-calling-v2-sharegpt ...")
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    results = []

    for row in ds:
        try:
            convs = row.get("conversations", [])
            tools_str = row.get("tools", "[]")

            # 解析 tools
            tools = []
            try:
                raw_tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw_tools, list):
                    for t in raw_tools:
                        tools.append(wrap_tool_openai(t))
            except (json.JSONDecodeError, TypeError):
                pass

            # 转换 conversations (与 glaive_zh 相同逻辑)
            messages = []
            if tools:
                messages.append({"role": "system", "content": "You are a helpful assistant with access to tools."})

            for conv in convs:
                role_from = conv.get("from", "")
                value = conv.get("value", "")

                if role_from == "human":
                    messages.append({"role": "user", "content": value})
                elif role_from == "gpt":
                    messages.append({"role": "assistant", "content": value})
                elif role_from == "function_call":
                    try:
                        fc = json.loads(value) if isinstance(value, str) else value
                        tc = make_tool_call(fc.get("name", "unknown"), fc.get("arguments", {}))
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tc]
                        })
                    except (json.JSONDecodeError, TypeError):
                        messages.append({"role": "assistant", "content": value})
                elif role_from == "observation":
                    tool_name = ""
                    for m in reversed(messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            tool_name = m["tool_calls"][0]["function"]["name"]
                            break
                    messages.append({
                        "role": "tool",
                        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                        "name": tool_name
                    })

            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools

            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> glaive_v2_sharegpt: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 4: NousResearch/hermes-function-calling-v1 (英文)
# ============================================================

def convert_hermes_fc(output_dir: Path) -> list:
    """转换 Hermes 函数调用数据集"""
    print("[4/7] Loading NousResearch/hermes-function-calling-v1 ...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    results = []

    tool_call_pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

    for row in ds:
        try:
            convs = row.get("conversations", [])
            tools_str = row.get("tools", "[]")

            # 解析 tools
            tools = []
            try:
                raw_tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw_tools, list):
                    for t in raw_tools:
                        tools.append(wrap_tool_openai(t))
            except (json.JSONDecodeError, TypeError):
                pass

            messages = []

            for conv in convs:
                role_from = conv.get("from", "")
                value = conv.get("value", "")

                if role_from == "system":
                    # 不使用原始系统提示（包含工具定义文本），用简洁版
                    if tools:
                        messages.append({"role": "system", "content": "You are a helpful assistant with access to tools."})
                    else:
                        messages.append({"role": "system", "content": value})
                elif role_from == "human":
                    messages.append({"role": "user", "content": value})
                elif role_from == "gpt":
                    # 检查是否包含 <tool_call> 标签
                    tc_matches = tool_call_pattern.findall(value)
                    if tc_matches:
                        tool_calls = []
                        for tc_json in tc_matches:
                            try:
                                tc_data = json.loads(tc_json)
                                tc = make_tool_call(
                                    tc_data.get("name", "unknown"),
                                    tc_data.get("arguments", {})
                                )
                                tool_calls.append(tc)
                            except json.JSONDecodeError:
                                continue

                        if tool_calls:
                            # 提取 tool_call 标签外的文本作为 content
                            remaining = tool_call_pattern.sub("", value).strip()
                            messages.append({
                                "role": "assistant",
                                "content": remaining if remaining else None,
                                "tool_calls": tool_calls
                            })
                        else:
                            messages.append({"role": "assistant", "content": value})
                    else:
                        messages.append({"role": "assistant", "content": value})

            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools

            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> hermes_fc: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 5: tryumanshow/ToolACE-Qwen-cleaned (英文)
# ============================================================

def convert_toolace_qwen(output_dir: Path) -> list:
    """转换 ToolACE-Qwen-cleaned 数据集"""
    print("[5/7] Loading tryumanshow/ToolACE-Qwen-cleaned ...")
    ds = load_dataset("tryumanshow/ToolACE-Qwen-cleaned", split="train")
    results = []

    for row in ds:
        try:
            tools_str = row.get("tools", "[]")
            convs_str = row.get("conversations", "[]")

            # 解析 tools (JSON string)
            tools = []
            try:
                raw_tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
                if isinstance(raw_tools, list):
                    for t in raw_tools:
                        tools.append(wrap_tool_openai(t))
            except (json.JSONDecodeError, TypeError):
                pass

            # 解析 conversations (JSON string)
            try:
                convs = json.loads(convs_str) if isinstance(convs_str, str) else convs_str
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(convs, list):
                continue

            messages = []
            if tools:
                messages.append({"role": "system", "content": "You are a helpful assistant with access to tools."})

            for conv in convs:
                role = conv.get("role", "")
                content = conv.get("content", "")

                if role == "user":
                    messages.append({"role": "user", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
                elif role == "assistant":
                    tc_raw = conv.get("tool_calls", [])
                    if tc_raw:
                        tool_calls = []
                        for tc in tc_raw:
                            func_data = tc.get("function", {})
                            name = func_data.get("name", "unknown")
                            args = func_data.get("arguments", {})
                            # arguments 可能是双重转义的 JSON 字符串
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {"raw": args}
                            tool_calls.append(make_tool_call(name, args))

                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                        })
                elif role == "tool":
                    # tool content 可能是 list/dict，需要转为字符串
                    tool_name = conv.get("name", "")
                    if not tool_name:
                        # 从上一条 assistant tool_calls 获取
                        for m in reversed(messages):
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                tool_name = m["tool_calls"][0]["function"]["name"]
                                break
                    tc_content = content
                    if not isinstance(tc_content, str):
                        tc_content = json.dumps(tc_content, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "content": tc_content,
                        "name": tool_name
                    })

            sample = {"messages": messages}
            if tools:
                sample["tools"] = tools

            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> toolace_qwen: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 6: nohurry/Opus-4.6-Reasoning (英文, 非工具调用)
# ============================================================

def convert_opus_reasoning(output_dir: Path) -> list:
    """转换 Opus 推理蒸馏数据集"""
    print("[6/7] Loading nohurry/Opus-4.6-Reasoning-3000x-filtered ...")
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")
    results = []

    for row in ds:
        try:
            problem = row.get("problem", "")
            thinking = row.get("thinking", "")
            solution = row.get("solution", "")

            if not problem or not solution:
                continue

            # 构建 assistant 回复: <think>推理过程</think>\n\n最终答案
            if thinking:
                assistant_content = f"<think>\n{thinking}\n</think>\n\n{solution}"
            else:
                assistant_content = solution

            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant_content}
            ]

            sample = {"messages": messages}
            if validate_sample(sample):
                results.append(sample)

        except Exception as e:
            continue

    print(f"  -> opus_reasoning: {len(results)} samples converted")
    return results


# ============================================================
# 转换器 7: bellfire/openclaw-coder-dataset (英文)
# ============================================================

def convert_openclaw(output_dir: Path) -> list:
    """转换 OpenClaw 编码助手数据集"""
    print("[7/7] Loading bellfire/openclaw-coder-dataset ...")

    results = []
    for split_name in ["train", "test"]:
        try:
            ds = load_dataset("bellfire/openclaw-coder-dataset", split=split_name)
        except Exception:
            continue

        for row in ds:
            try:
                raw_messages = row.get("messages", [])
                if not raw_messages:
                    continue

                messages = []
                for m in raw_messages:
                    role = m.get("role", "")
                    content = m.get("content", "")

                    if role == "system":
                        messages.append({"role": "system", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
                    elif role == "user":
                        messages.append({"role": "user", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
                    elif role == "assistant":
                        tc_raw = m.get("tool_calls", [])
                        if tc_raw:
                            tool_calls = []
                            for tc in tc_raw:
                                func_data = tc.get("function", {})
                                name = func_data.get("name", "unknown")
                                args = func_data.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {"raw": args}
                                tool_calls.append(make_tool_call(name, args))
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                            })
                    elif role == "tool":
                        tool_name = m.get("name", "")
                        if not tool_name:
                            for prev in reversed(messages):
                                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                                    tool_name = prev["tool_calls"][0]["function"]["name"]
                                    break
                        tc_content = content
                        if not isinstance(tc_content, str):
                            tc_content = json.dumps(tc_content, ensure_ascii=False)
                        messages.append({
                            "role": "tool",
                            "content": tc_content,
                            "name": tool_name
                        })

                sample = {"messages": messages}
                # OpenClaw 没有单独的 tools 字段
                if validate_sample(sample):
                    results.append(sample)

            except Exception as e:
                continue

    print(f"  -> openclaw: {len(results)} samples converted")
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    output_dir = Path("/Users/icecee/qwen35-finetune/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen3.5-9B Tool-Calling SFT 数据准备")
    print("=" * 60)

    # 收集所有数据
    all_samples = []

    converters = [
        ("Deepexi (中文)", convert_deepexi),
        ("glaive_zh (中文)", convert_glaive_zh),
        ("glaive_v2_sharegpt (英文)", convert_glaive_v2_sharegpt),
        ("hermes_fc (英文)", convert_hermes_fc),
        ("toolace_qwen (英文)", convert_toolace_qwen),
        ("opus_reasoning (英文)", convert_opus_reasoning),
        ("openclaw (英文)", convert_openclaw),
    ]

    stats = {}
    for name, converter in converters:
        try:
            samples = converter(output_dir)
            stats[name] = len(samples)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            stats[name] = 0

    print(f"\n总计转换: {len(all_samples)} 样本")

    # 去重
    print("\n去重中...")
    seen = set()
    deduped = []
    for s in all_samples:
        h = msg_hash(s["messages"])
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    print(f"  去重前: {len(all_samples)}, 去重后: {len(deduped)}, 移除: {len(all_samples) - len(deduped)}")

    # Shuffle
    print("打乱数据...")
    random.seed(42)
    random.shuffle(deduped)

    # 划分 train / valid (90% / 10%)
    split_idx = int(len(deduped) * 0.9)
    train_data = deduped[:split_idx]
    valid_data = deduped[split_idx:]

    # 写入 JSONL
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    print(f"\n写入训练集: {train_path} ({len(train_data)} 样本)")
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"写入验证集: {valid_path} ({len(valid_data)} 样本)")
    with open(valid_path, "w", encoding="utf-8") as f:
        for sample in valid_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计摘要
    print("\n" + "=" * 60)
    print("数据集统计摘要")
    print("=" * 60)
    for name, count in stats.items():
        print(f"  {name:40s}: {count:>8,}")
    print(f"  {'总计 (去重后)':40s}: {len(deduped):>8,}")
    print(f"  {'训练集':40s}: {len(train_data):>8,}")
    print(f"  {'验证集':40s}: {len(valid_data):>8,}")

    # 统计有 tools 的样本比例
    with_tools = sum(1 for s in deduped if "tools" in s and s["tools"])
    without_tools = len(deduped) - with_tools
    if len(deduped) > 0:
        print(f"\n  有工具定义的样本: {with_tools:>8,} ({with_tools/len(deduped)*100:.1f}%)")
        print(f"  无工具定义的样本: {without_tools:>8,} ({without_tools/len(deduped)*100:.1f}%)")
    else:
        print("\n  警告: 没有成功转换任何样本!")

    # 文件大小
    train_size = train_path.stat().st_size / (1024 * 1024)
    valid_size = valid_path.stat().st_size / (1024 * 1024)
    print(f"\n  训练集文件大小: {train_size:.1f} MB")
    print(f"  验证集文件大小: {valid_size:.1f} MB")
    print("=" * 60)
    print("数据准备完成!")


if __name__ == "__main__":
    main()
