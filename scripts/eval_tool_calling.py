#!/usr/bin/env python3
"""
测试微调后模型的工具调用能力

用法:
  python eval_tool_calling.py --model_dir ./output/merged_bf16
  python eval_tool_calling.py --model_dir ./output/final_adapter --base_model Qwen/Qwen3.5-9B-Base
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEST_CASES = [
    # 中文工具调用
    {
        "name": "中文-天气查询",
        "messages": [
            {"role": "user", "content": "请帮我查询北京今天的天气情况"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气信息",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"}
                }, "required": ["city"]}
            }}
        ],
    },
    {
        "name": "中文-计算器",
        "messages": [
            {"role": "user", "content": "帮我计算一下 123 乘以 456 等于多少"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {"type": "object", "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                }, "required": ["expression"]}
            }}
        ],
    },
    # 英文工具调用
    {
        "name": "EN-File Search",
        "messages": [
            {"role": "user", "content": "Search for all Python files in the src directory"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "search_files",
                "description": "Search for files matching a pattern in a directory",
                "parameters": {"type": "object", "properties": {
                    "directory": {"type": "string", "description": "Directory to search in"},
                    "pattern": {"type": "string", "description": "File name pattern (glob)"},
                    "recursive": {"type": "boolean", "description": "Search recursively"}
                }, "required": ["directory", "pattern"]}
            }}
        ],
    },
    {
        "name": "EN-Multi-tool",
        "messages": [
            {"role": "user",
             "content": "I need to know the weather in Tokyo and also translate 'hello world' to Japanese"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get weather information for a city",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"},
                }, "required": ["city"]}
            }},
            {"type": "function", "function": {
                "name": "translate",
                "description": "Translate text to a target language",
                "parameters": {"type": "object", "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {"type": "string", "description": "Target language code"}
                }, "required": ["text", "target_language"]}
            }},
        ],
    },
    # 无工具 — 应该正常回答
    {
        "name": "中文-普通对话",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "tools": None,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None,
                        help="如果 model_dir 是 adapter，指定 base model")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    print("加载模型...")
    if args.base_model:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model.eval()
    print(f"模型已加载: {type(model).__name__}")
    print()

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"{'=' * 60}")
        print(f"测试 {i}/{len(TEST_CASES)}: {tc['name']}")
        print(f"{'=' * 60}")
        print(f"User: {tc['messages'][0]['content']}")
        if tc['tools']:
            tool_names = [t['function']['name'] for t in tc['tools']]
            print(f"Tools: {', '.join(tool_names)}")
        print()

        input_text = tokenizer.apply_chat_template(
            tc["messages"],
            tools=tc["tools"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        # 截断到 <|im_end|>
        if "<|im_end|>" in response:
            response = response[:response.index("<|im_end|>")]

        print(f"Assistant: {response.strip()}")
        print()


if __name__ == "__main__":
    main()
