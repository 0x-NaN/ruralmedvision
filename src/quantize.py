"""
quantize.py -- RuralMed Vision Post-Training Quantization

Merges LoRA adapter into base model and exports as GGUF (4-bit) for
fully offline deployment via llama.cpp.

This is the key step for the rural deployment story:
  Fine-tuned 7B model -> 4-bit GGUF -> runs on budget laptop, no internet

Usage:
    python src/quantize.py --config configs/config.yaml

Requirements:
    pip install llama-cpp-python
    # llama.cpp must be installed separately for GGUF conversion:
    # git clone https://github.com/ggerganov/llama.cpp
    # cd llama.cpp && make
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def merge_adapter(cfg: dict) -> str:
    """Merge LoRA adapter into base model and save full merged model."""
    base_model_id = cfg["model"]["name"]
    adapter_path  = cfg["model"]["adapter_path"]
    merged_path   = str(Path(adapter_path).parent / "ruralmed_merged")

    print(f"Loading base model: {base_model_id}")
    # Load in fp16 for merging (not quantized -- need full weights to merge)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",           # merge on CPU to avoid VRAM limits
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model -> {merged_path}")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(merged_path)

    print(f"Merged model saved -> {merged_path}")
    return merged_path


def convert_to_gguf(merged_path: str, output_path: str):
    """
    Convert merged HuggingFace model to GGUF format using llama.cpp.
    Requires llama.cpp to be cloned and built in the system.
    """
    # Try to find llama.cpp convert script
    candidates = [
        "./llama.cpp/convert_hf_to_gguf.py",
        "../llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]

    convert_script = None
    for c in candidates:
        if Path(c).exists():
            convert_script = c
            break

    if convert_script is None:
        print("\nllama.cpp not found. To convert to GGUF:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make")
        print(f"  python convert_hf_to_gguf.py {merged_path} --outtype q4_k_m --outfile {output_path}")
        print("\nSkipping GGUF conversion -- merged model saved for manual conversion.")
        return False

    import subprocess
    cmd = [
        sys.executable, convert_script,
        merged_path,
        "--outtype", "q4_k_m",      # 4-bit quantization
        "--outfile", output_path,
    ]
    print(f"Converting to GGUF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"GGUF model saved -> {output_path}")
        size_gb = Path(output_path).stat().st_size / 1e9
        print(f"File size: {size_gb:.2f} GB")
        return True
    else:
        print(f"Conversion failed:\n{result.stderr}")
        return False


def benchmark_gguf(gguf_path: str):
    """Quick benchmark of GGUF inference speed."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Skipping benchmark.")
        print("Install with: pip install llama-cpp-python")
        return

    import time
    print(f"\nBenchmarking GGUF inference...")
    llm = Llama(model_path=gguf_path, n_ctx=512, n_gpu_layers=0)  # CPU only for benchmark

    prompt = "Patient presents with a dark irregular skin lesion. Provide a triage assessment."
    start  = time.perf_counter()
    out    = llm(prompt, max_tokens=100, echo=False)
    elapsed = time.perf_counter() - start

    tokens_generated = out["usage"]["completion_tokens"]
    print(f"Tokens generated : {tokens_generated}")
    print(f"Time             : {elapsed:.2f}s")
    print(f"Speed            : {tokens_generated/elapsed:.1f} tok/s (CPU)")
    print(f"\nSample output:\n{out['choices'][0]['text']}")


def main():
    parser = argparse.ArgumentParser(description="RuralMed Vision quantization")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Skip GGUF conversion, just merge adapter")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark GGUF inference speed after conversion")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=== RuralMed Vision -- Post-Training Quantization ===\n")

    # Step 1: Merge adapter
    merged_path = merge_adapter(cfg)

    if args.skip_gguf:
        print("\nSkipped GGUF conversion (--skip-gguf flag).")
        print(f"Merged model at: {merged_path}")
        return

    # Step 2: Convert to GGUF
    gguf_path = cfg["model"]["quantized_path"]
    Path(gguf_path).parent.mkdir(parents=True, exist_ok=True)
    success = convert_to_gguf(merged_path, gguf_path)

    # Step 3: Benchmark
    if success and args.benchmark:
        benchmark_gguf(gguf_path)

    print("\nQuantization complete.")
    print("\nRural deployment summary:")
    print(f"  Original 7B model : ~14 GB (bf16)")
    print(f"  4-bit GGUF        : ~4.5 GB (q4_k_m)")
    print(f"  Runs on           : any laptop with 8GB RAM, no GPU required")
    print(f"  Internet needed   : No (fully offline after download)")


if __name__ == "__main__":
    main()
