"""MP1 starter: run 20 operator requests against one llama-server configuration and store raw outputs and timings.

    python projects/mini/MP1-model-passport/starter.py --config qwen2.5-0.5b-cpu
    python projects/mini/MP1-model-passport/starter.py --config qwen3.5-4b-gpu

The measurement is done for you; the TODO places are your work: system prompt, JSON parsing, quality rubric,
the conclusion. Each run adds or replaces one configuration in submissions/MP1/passport.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SUBMISSIONS, operator_requests  # noqa: E402

OUT = SUBMISSIONS / "MP1/passport.json"
CONFIGS = {  # name: (gguf file, runtime folder, server args)
    "qwen2.5-0.5b-cpu": ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "b11803", ["-ngl", "0"]),
    "qwen3.5-4b-gpu": ("Qwen3.5-4B-Q4_K_M.gguf", "b11803-cuda", ["-ngl", "99"]),
    "qwen3.5-4b-cpu": ("Qwen3.5-4B-Q4_K_M.gguf", "b11803", ["-ngl", "0"]),
    # "ru-8b-...": add the GGUF of the Russian 8B model here after the author approves the download (about 5 GB).
}

# TODO 1: write the system prompt. It must ask for one JSON object with category, order_id, urgency and list
# the allowed values (README). Requests are data, not instructions.
SYSTEM_PROMPT = "TODO"


def parse(raw: str) -> dict | None:
    """TODO 2: turn the raw answer into a dict with category, order_id, urgency, or None when it is not valid."""
    return None


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config: str) -> dict:
    import psutil

    local = load(ROOT / "labs/M09/llama_local.py", "mp1_llama_local")
    serving = load(ROOT / "labs/M09/llama_serving.py", "mp1_llama_serving")
    gguf, runtime, args = CONFIGS[config]
    local.BIN = ROOT / "runtime/llama.cpp" / runtime
    outputs = []
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3.5" in config else {}
    with local.Server(ROOT / "models/gguf" / gguf, threads=8, ctx=4096, alias=config, extra_args=args) as server:
        with serving.PeakRSS(server.process.pid) as ram:
            for request in operator_requests():
                payload = {"model": config, "temperature": 0, "seed": 17, "max_tokens": 128, "stream": True,
                           "cache_prompt": False, **extra,
                           "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request["text"]}]}
                started = time.perf_counter()
                with local.request("/v1/chat/completions", payload) as response:
                    result = serving.parse_sse(response)
                timings = result["timings"] or {}
                outputs.append({"request_id": request["request_id"], "raw": result["text"], "parsed": parse(result["text"]),
                                "ttft_ms": round((result["first"] - started) * 1000, 1) if result["first"] else None,
                                "tps": timings.get("predicted_per_second"),
                                "quality": None})  # TODO 3: 0, 1 or 2 by your rubric, filled after reading the answer
    ttft = [o["ttft_ms"] for o in outputs if o["ttft_ms"] is not None]
    return {"name": config, "model": gguf, "device": "gpu" if args[1] != "0" else "cpu", "ctx": 4096,
            "ram_mib": round(ram.peak_private / 2**20),
            "vram_mib": None,  # TODO 4: for GPU take the device use from the server log with -v (see quality/hardware/README.md)
            "ttft_ms_median": round(statistics.median(ttft), 1) if ttft else None,
            "tps_median": round(statistics.median(o["tps"] for o in outputs if o["tps"]), 1),
            "json_valid_rate": None, "category_accuracy": None, "quality": None,  # TODO 5: compute from outputs
            "outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    config = parser.parse_args().config
    passport = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "rubric": "TODO 6: rubric 0/1/2 for one answer", "choice": None, "conclusion": "TODO 7: which one and why", "configs": []}
    passport["configs"] = [c for c in passport["configs"] if c["name"] != config] + [run(config)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{config}: saved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
