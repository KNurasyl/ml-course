"""Local LLM serving with llama.cpp: GGUF, quantization, OpenAI-compatible API, speed and memory on CPU
(extension 1.2.0, T30, lesson M09-L07X).

Configurations, fixed before running: Qwen2.5-0.5B-Instruct in transformers float32 (the course model), the same model
as GGUF Q8_0 and Q4_K_M in llama-server, and Qwen3.5-4B Q4_K_M (community quantization by unsloth; no official GGUF).
All use 4 CPU threads, the same six prompts, greedy decoding and at most 64 new tokens. Qwen3.5 thinking is switched off
per request with chat_template_kwargs, as its model card describes. Measured per configuration: load time, time to the
first token over a streamed response, prefill and decode tokens per second, peak RSS of the serving process sampled every
30 ms, file size. Answers of the 0.5B runs are compared with its float32 output: identical text and common prefix.
This is not a quality benchmark: T51 compares the models on rubric-graded answers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M09/llama"
THREADS, MAX_TOKENS, CTX = 4, 64, 4096
PROMPTS = [
    "Объясни в двух предложениях, что такое индекс в базе данных.",
    "What does an idempotent HTTP operation mean? Answer in two sentences.",
    "Назови три причины, по которым запрос к API может выполняться медленно.",
    "Чем отличается транзакция от отдельного SQL-запроса? Ответь кратко.",
    "Explain the difference between p50 and p95 latency in one short paragraph.",
    "Сформулируй один вопрос для собеседования backend-разработчика о кэшировании.",
]
CACHE_PROMPT = "Объясни в двух предложениях, зачем сервису кэш ответов и когда он вредит."
CONFIGS = [  # name, runtime, registry entry
    ("qwen2.5-0.5b-f32-transformers", "transformers", None),
    ("qwen2.5-0.5b-q8_0-llama.cpp", "llama.cpp", "qwen2.5-0.5b-instruct-q8_0"),
    ("qwen2.5-0.5b-q4_k_m-llama.cpp", "llama.cpp", "qwen2.5-0.5b-instruct-q4_k_m"),
    ("qwen3.5-4b-q4_k_m-llama.cpp", "llama.cpp", "qwen3.5-4b-q4_k_m"),
]


def load(name):
    spec = importlib.util.spec_from_file_location(f"m09_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry():
    return json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))


# ggml tensor types by id (ggml/include/ggml.h of the pinned llama.cpp release); ids 4, 5, 31-33, 36-38 were removed.
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
              20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
              29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
META_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_summary(path):
    """Read a GGUF header (ggml docs/gguf.md): chosen metadata and, per tensor type, tensors, weights, bytes, bits per weight.
    A tensor's size is the distance to the next tensor's offset, so it includes alignment padding (tiny for real tensors)."""
    path = Path(path)
    with path.open("rb") as f:
        u32 = lambda: struct.unpack("<I", f.read(4))[0]
        u64 = lambda: struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf-8", "replace")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = u32(), u64()
                return [value(inner) for _ in range(count)]
            return struct.unpack(META_SCALARS[kind], f.read(struct.calcsize(META_SCALARS[kind])))[0]

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path.name} is not a GGUF file")
        version, n_tensors, n_meta = u32(), u64(), u64()
        meta = {}
        for _ in range(n_meta):
            key = string()
            kind = u32()
            item = value(kind)
            meta[key] = len(item) if isinstance(item, list) else item  # token lists: keep only their length
        tensors = []
        for _ in range(n_tensors):
            name = string()
            dims = [u64() for _ in range(u32())]
            tensors.append({"name": name, "weights": int(np.prod(dims)), "type": GGML_TYPES.get(u32(), "unknown"), "offset": u64()})
        alignment = int(meta.get("general.alignment", 32))
        data_start = f.tell() + (alignment - f.tell() % alignment) % alignment
    ends = sorted(t["offset"] for t in tensors)[1:] + [path.stat().st_size - data_start]
    size = dict(zip(sorted(t["offset"] for t in tensors), ends))
    types = {}
    for t in tensors:
        row = types.setdefault(t["type"], {"tensors": 0, "weights": 0, "bytes": 0})
        row["tensors"] += 1
        row["weights"] += t["weights"]
        row["bytes"] += size[t["offset"]] - t["offset"]
    for row in types.values():
        row["bits_per_weight"] = round(row["bytes"] * 8 / row["weights"], 2)
    arch = meta.get("general.architecture", "")
    weights, total = sum(r["weights"] for r in types.values()), sum(r["bytes"] for r in types.values())
    return {"version": version, "architecture": arch, "name": meta.get("general.name"), "file_type": meta.get("general.file_type"),
            "context_length": meta.get(f"{arch}.context_length"), "layers": meta.get(f"{arch}.block_count"),
            "embedding": meta.get(f"{arch}.embedding_length"), "has_chat_template": "tokenizer.chat_template" in meta,
            "tensors": len(tensors), "weights": weights, "tensor_bytes": total, "bits_per_weight": round(total * 8 / weights, 2),
            "types": dict(sorted(types.items(), key=lambda kv: -kv[1]["bytes"]))}


def gguf_entry(name):
    return next(e for e in registry()["gguf"] if e["name"] == name)


def parse_sse(lines, clock=time.perf_counter):
    """Read an OpenAI-style server-sent event stream: answer text, time of the first content, llama.cpp timings."""
    parts, first, timings, usage, reasoning = [], None, None, None, 0
    for raw in lines:
        line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        timings = chunk.get("timings") or timings
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning += len(delta.get("reasoning_content") or "")
            if delta.get("content"):
                first = first if first is not None else clock()
                parts.append(delta["content"])
    return {"text": "".join(parts), "first": first, "timings": timings, "usage": usage, "reasoning_chars": reasoning}


def chat_stream(local, prompt, extra=None, model="local", cache_prompt=False):
    # cache_prompt is a llama-server field: True reuses the KV cache of a shared prompt prefix from the previous request.
    # Measurements switch it off, so every prompt is prefilled from scratch, as in transformers.
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
               "temperature": 0, "seed": 17, "stream": True, "stream_options": {"include_usage": True},
               "cache_prompt": cache_prompt, **(extra or {})}
    started = time.perf_counter()
    with local.request("/v1/chat/completions", payload) as response:
        result = parse_sse(response)
    ended = time.perf_counter()
    timings = result.pop("timings") or {}
    first = result.pop("first")
    return {**result, "ttft_ms": None if first is None else (first - started) * 1000, "total_ms": (ended - started) * 1000,
            "prompt_tokens": timings.get("prompt_n"), "prefill_tps": timings.get("prompt_per_second"),
            "generated_tokens": timings.get("predicted_n"), "decode_tps": timings.get("predicted_per_second")}


def memory(processes):
    """RSS counts file pages mapped with mmap, which the OS may drop; private bytes (Windows) are the process's own memory."""
    rss = private = 0
    for process in processes:
        try:
            info = process.memory_info()
        except psutil.NoSuchProcess:
            continue
        rss += info.rss
        private += getattr(info, "private", 0)
    return rss, private


class PeakRSS:
    """Sample the memory of a process tree every 30 ms; the venv launcher may spawn the real interpreter."""

    def __init__(self, pid):
        self.process, self.peak, self.peak_private, self.running = psutil.Process(pid), 0, 0, True
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while self.running:
            try:
                rss, private = memory([self.process, *self.process.children(recursive=True)])
            except psutil.NoSuchProcess:
                return
            self.peak, self.peak_private = max(self.peak, rss), max(self.peak_private, private)
            time.sleep(0.03)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()


def run_llama(name, entry_name, threads):
    local = load("llama_local")
    entry = gguf_entry(entry_name)
    model = ROOT / entry["path"] / entry["file"]
    if not model.exists():
        return {"status": "not_downloaded", "file": entry["file"]}
    try:
        local.server_exe()
    except FileNotFoundError:
        return {"status": "llama_cpp_missing"}
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if entry_name.startswith("qwen3.5") else None
    with local.Server(model, threads=threads, ctx=CTX, alias=entry_name) as server:
        idle, idle_private = memory([psutil.Process(server.process.pid)])
        with local.request("/v1/models") as response:
            models = [m["id"] for m in json.load(response)["data"]]
        chat_stream(local, "Hello", extra, entry_name)  # warm-up, not measured
        with PeakRSS(server.process.pid) as sampler:
            answers = [{"prompt": p, **chat_stream(local, p, extra, entry_name)} for p in PROMPTS]
        # A prompt the server has not seen, twice, with prefix caching: the first request reuses only the shared
        # system part of the chat template, the repeat reuses the KV cache of the whole prompt.
        cached = [chat_stream(local, CACHE_PROMPT, extra, entry_name, cache_prompt=True) for _ in range(2)]
        load_seconds = server.load_seconds
    return {"status": "measured", "runtime": "llama.cpp", "file": entry["file"], "file_bytes": model.stat().st_size,
            "repo_id": entry["repo_id"], "revision": entry["revision"], "license": entry["license"], "source": entry["source"],
            "load_seconds": round(load_seconds, 2), "rss_idle_bytes": idle, "rss_peak_bytes": sampler.peak,
            "private_idle_bytes": idle_private, "private_peak_bytes": sampler.peak_private,
            "gguf": gguf_summary(model),
            "v1_models": models, "thinking": "disabled per request" if extra else "not applicable", "answers": answers,
            "prompt_cache": [{"ttft_ms": round(c["ttft_ms"], 1), "prompt_tokens_computed": c["prompt_tokens"]} for c in cached]}


def transformers_worker(out_path, threads):
    """Child process: the course model in float32 through transformers, greedy, same prompts and token limit."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    entry = registry()["llm"]
    path = ROOT / entry["path"]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32)
    model.eval()
    load_seconds = time.perf_counter() - started
    idle, idle_private = memory([psutil.Process()])

    def generate(prompt, new_tokens):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return inputs["input_ids"].shape[1], output[0, inputs["input_ids"].shape[1]:], (time.perf_counter() - started) * 1000

    generate("Hello", 4)  # warm-up, not measured
    answers = []
    for prompt in PROMPTS:
        prompt_tokens, _, first_ms = generate(prompt, 1)  # prefill and the first token
        _, tokens, total_ms = generate(prompt, MAX_TOKENS)
        n = int(tokens.shape[0])
        answers.append({"prompt": prompt, "text": tokenizer.decode(tokens, skip_special_tokens=True), "ttft_ms": first_ms,
                        "total_ms": total_ms, "prompt_tokens": int(prompt_tokens), "prefill_tps": prompt_tokens / first_ms * 1000,
                        "generated_tokens": n, "decode_tps": (n - 1) / max(total_ms - first_ms, 1e-9) * 1000 if n > 1 else None})
    weights = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    Path(out_path).write_text(json.dumps({"status": "measured", "runtime": "transformers float32", "file_bytes": weights,
                                          "repo_id": entry["repo_id"], "revision": entry["revision"], "license": entry["license"],
                                          "load_seconds": round(load_seconds, 2), "rss_idle_bytes": idle,
                                          "private_idle_bytes": idle_private, "answers": answers},
                                         ensure_ascii=False), encoding="utf-8")


def run_transformers(threads):
    if not any((ROOT / registry()["llm"]["path"]).glob("*.safetensors")):
        return {"status": "not_downloaded"}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "transformers-worker.json"
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(out_path), "--threads", str(threads)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with PeakRSS(child.pid) as sampler:
        child.wait()
    if child.returncode:
        raise RuntimeError("transformers worker failed")
    result = json.loads(out_path.read_text(encoding="utf-8"))
    out_path.unlink()
    return {**result, "rss_peak_bytes": sampler.peak, "private_peak_bytes": sampler.peak_private,
            "rss_note": "whole Python process with PyTorch, sampled from outside"}


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def summarize(result, reference=None):
    answers = result["answers"]
    med = lambda key: round(statistics.median([a[key] for a in answers if a.get(key) is not None]), 1)
    summary = {"ttft_ms_median": med("ttft_ms"), "prefill_tps_median": med("prefill_tps"), "decode_tps_median": med("decode_tps"),
               "generated_tokens_total": sum(a["generated_tokens"] or 0 for a in answers)}
    if reference:
        pairs = list(zip(reference["answers"], answers))
        summary["same_text_as_f32"] = sum(r["text"] == a["text"] for r, a in pairs)
        summary["common_prefix_chars_median"] = statistics.median(common_prefix(r["text"], a["text"]) for r, a in pairs)
    return summary


def run(only=None, threads=THREADS, out_dir=None):
    results = {}
    for name, runtime, entry in CONFIGS:
        if only and name not in only:
            continue
        results[name] = run_transformers(threads) if runtime == "transformers" else run_llama(name, entry, threads)
    reference = results.get(CONFIGS[0][0])
    reference = reference if reference and reference.get("status") == "measured" else None
    for name, result in results.items():
        if result.get("status") == "measured":
            same_model = name.startswith("qwen2.5-0.5b") and name != CONFIGS[0][0]
            result["summary"] = summarize(result, reference if same_model else None)
    local = load("llama_local")
    report = {"task": "T30", "llama_cpp": local.RELEASE, "threads": threads, "max_tokens": MAX_TOKENS, "ctx": CTX,
              "decoding": "greedy (temperature 0), seed 17", "prompts": PROMPTS,
              "fixed_before_running": "configurations, prompts, 4 threads, 64 new tokens, ctx 4096, greedy decoding",
              "results": results,
              "limits": "One machine, CPU, six short prompts; RSS of a native server and of a Python process are not the same "
                        "kind of number. Speed and memory, not answer quality."}
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def table_text(report):
    lines = ["| Конфигурация | Файл, МБ | Загрузка, с | TTFT, мс | Prefill, ток/с | Генерация, ток/с | Пик RSS / private, МБ | Совпало с f32 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in report["results"].items():
        if r.get("status") != "measured":
            lines.append(f"| {name} | {r.get('status')} | | | | | | |")
            continue
        s = r["summary"]
        same = f"{s['same_text_as_f32']} из {len(r['answers'])}" if "same_text_as_f32" in s else "—"
        lines.append(f"| {name} | {r['file_bytes'] / 2**20:.0f} | {r['load_seconds']} | {s['ttft_ms_median']} | {s['prefill_tps_median']} | "
                     f"{s['decode_tps_median']} | {r['rss_peak_bytes'] / 2**20:.0f} / {r['private_peak_bytes'] / 2**20:.0f} | {same} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=[c[0] for c in CONFIGS])
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        transformers_worker(args.worker, args.threads)
    else:
        report = run(args.only, args.threads)
        print(table_text(report))
