#!/usr/bin/env python3
"""
训练完成后导出模型为 GGUF 格式
用于 Ollama / LM Studio / llama.cpp 部署

用法:
  python export_gguf.py --model_dir ./output/merged_bf16
  python export_gguf.py --model_dir ./output/merged_bf16 --quantization Q4_K_M
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="导出模型为 GGUF 格式")
    parser.add_argument("--model_dir", type=str, required=True, help="合并后模型路径")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF 量化类型")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="GGUF 输出目录 (默认: model_dir/gguf)")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    output_dir = args.output_dir or os.path.join(args.model_dir, "..", "gguf")
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载模型: {args.model_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=4096,
        load_in_16bit=True,
    )

    print(f"导出 GGUF ({args.quantization})...")
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=args.quantization,
    )
    print(f"GGUF 已导出: {output_dir}")

    # 生成 Ollama Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
    if gguf_files:
        gguf_name = gguf_files[0]
        with open(modelfile_path, "w") as f:
            f.write(f'FROM ./{gguf_name}\n')
            f.write('PARAMETER temperature 0.7\n')
            f.write('PARAMETER top_p 0.9\n')
            f.write('PARAMETER stop "<|im_end|>"\n')
            f.write('PARAMETER stop "<|endoftext|>"\n')
        print(f"Ollama Modelfile 已生成: {modelfile_path}")
        print()
        print("导入 Ollama:")
        print(f"  cd {output_dir}")
        print(f"  ollama create qwen35-tool-calling -f Modelfile")
        print(f"  ollama run qwen35-tool-calling")


if __name__ == "__main__":
    main()
