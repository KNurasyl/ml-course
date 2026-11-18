"""4-bit loading check for R06; runs only in .venv-gpu (torch with CUDA, bitsandbytes), never in the main .venv.

Loads the local Qwen2.5-0.5B-Instruct twice on the GPU, in float16 and in 4-bit NF4 (bitsandbytes, double quantization,
float16 compute), generates 32 greedy tokens with each and prints one JSON line: versions, memory footprint of the
weights, peak allocated CUDA memory and decode speed. Called by tools/hardware_baseline.py bench.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"
PROMPT = "Объясни в двух предложениях, что такое индекс в базе данных."
NEW_TOKENS = 32


def run(quantization):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kwargs = {"dtype": torch.float16, "device_map": {"": 0}}
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    load_seconds = time.perf_counter() - started
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(0)
    model.generate(**inputs, max_new_tokens=4, do_sample=False)  # warm-up
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = output.shape[1] - inputs["input_ids"].shape[1]
    result = {"quantization": quantization, "load_seconds": round(load_seconds, 2),
              "weights_mib": round(model.get_memory_footprint() / 2**20),
              "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
              "generated_tokens": int(generated), "tokens_per_second": round(generated / seconds, 1),
              "answer": tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:200],
              "linear_4bit_layers": sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")}
    del model
    return result


def main() -> int:
    import bitsandbytes
    import torch
    import transformers

    report = {"status": "failed", "python": platform.python_version(), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "bitsandbytes": bitsandbytes.__version__, "transformers": transformers.__version__,
              "model": "Qwen2.5-0.5B-Instruct (models/, safetensors)", "wsl": False}
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
        report["runs"] = [run("float16"), run("nf4")]
        report["status"] = "ok" if report["runs"][1]["linear_4bit_layers"] > 0 else "not_quantized"
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
