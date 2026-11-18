"""Hardware baseline of the learner's laptop (REFACTORING_PLAN_2.0, R06): llama.cpp on CPU and GPU, 4-bit PyTorch.

    python tools/hardware_baseline.py fetch-cuda      pinned CUDA build of llama.cpp b11803 into runtime/llama.cpp
    python tools/hardware_baseline.py bench           measure every configuration, write quality/hardware/baseline.json
    python tools/hardware_baseline.py bench --only qwen2.5-0.5b-q4_k_m-gpu

Per configuration: load time, time to the first token over a streamed response (median of six prompts), prefill and
decode tokens per second from llama.cpp timings, peak RAM of the server process (RSS and private bytes), peak VRAM
of the server process (nvidia-smi, sampled about twice a second) and the layer offload reported in the server log.
The prompts and decoding settings are those of labs/M09/llama_serving.py (T30), so the CPU rows are comparable.

The 4-bit check runs tools/hardware_4bit.py in the separate .venv-gpu (torch with CUDA, bitsandbytes); the main
.venv keeps the CPU torch the tests are pinned to. Nothing is installed system-wide.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.util
import json
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "quality/hardware/baseline.json"
GPU_PYTHON = ROOT / ".venv-gpu/Scripts/python.exe"
TAG = "b11803"
CUDA_DIR = ROOT / "runtime/llama.cpp" / f"{TAG}-cuda"
# Release b11803 (21.10.2026), SHA-256 from the GitHub release asset digests, checked 18.11.2026.
CUDA_ASSETS = [
    ("llama-b11803-bin-win-cuda-13.4-x64.zip", "04b36b76c3708a6ce6120a45011f880054a179b6d9906ad54b1a8eb79cbb3d8b"),
    ("cudart-llama-bin-win-cuda-13.4-x64.zip", "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"),
]
CPU_THREADS = 8  # i7-11800H: 8 physical cores; T30 used 4 for a course-wide comparison
CONFIGS = [  # name, registry entry, device, context, extra server args
    ("qwen2.5-0.5b-q4_k_m-cpu", "qwen2.5-0.5b-instruct-q4_k_m", "cpu", 4096, ["-ngl", "0"]),
    ("qwen2.5-0.5b-q4_k_m-gpu", "qwen2.5-0.5b-instruct-q4_k_m", "gpu", 4096, ["-ngl", "99"]),
    ("qwen3.5-4b-q4_k_m-cpu", "qwen3.5-4b-q4_k_m", "cpu", 4096, ["-ngl", "0"]),
    ("qwen3.5-4b-q4_k_m-gpu", "qwen3.5-4b-q4_k_m", "gpu", 4096, ["-ngl", "99"]),
    ("qwen3.5-4b-q4_k_m-gpu-ctx8k", "qwen3.5-4b-q4_k_m", "gpu", 8192, ["-ngl", "99"]),
    ("qwen3.5-4b-q4_k_m-gpu-ctx16k", "qwen3.5-4b-q4_k_m", "gpu", 16384, ["-ngl", "99"]),
]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_cuda() -> Path:
    download = CUDA_DIR.parent / "download"
    download.mkdir(parents=True, exist_ok=True)
    for name, digest in CUDA_ASSETS:
        archive = download / name
        if not archive.exists():
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{name}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive.with_suffix(".partial"))
            archive.with_suffix(".partial").replace(archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"sha256 mismatch for {name}: {actual}")
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if not (CUDA_DIR / member.filename).resolve().is_relative_to(CUDA_DIR.resolve()):
                    raise ValueError("Unsafe ZIP member")
            package.extractall(CUDA_DIR)
    print(f"llama.cpp {TAG} CUDA ready: {CUDA_DIR}")
    return CUDA_DIR


def nvidia(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout
    return [[cell.strip() for cell in line.split(",")] for line in out.strip().splitlines() if line.strip()]


class PeakVRAM:
    """Peak GPU memory of one process and of the whole card, sampled with nvidia-smi."""

    def __init__(self, pid: int):
        self.pid, self.process_mib, self.card_mib, self.running = pid, 0, 0, True
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        for pid, used in nvidia("pid,used_memory", "apps"):
            if pid.isdigit() and int(pid) == self.pid and used.isdigit():
                self.process_mib = max(self.process_mib, int(used))
        card = nvidia("memory.used")
        if card and card[0][0].isdigit():
            self.card_mib = max(self.card_mib, int(card[0][0]))

    def loop(self):
        while self.running:
            self.sample()
            time.sleep(0.3)

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()
        self.sample()


def offload(text: str) -> dict:
    """Layer offload and the device memory breakdown that llama-server prints with -v."""
    layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    free = re.search(r"CUDA0\s*:.*\((\d+) MiB, (\d+) MiB free\)", text)
    breakdown = re.search(r"CUDA0 \(.*?\)\s*\|\s*(\d+) = (\d+) \+ \((\d+) =\s*(\d+) \+\s*(\d+) \+\s*(\d+)\)", text)
    projected = re.search(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory", text)
    return {"layers_on_gpu": [int(layers.group(1)), int(layers.group(2))] if layers else None,
            "device_total_mib": int(free.group(1)) if free else None, "device_free_before_load_mib": int(free.group(2)) if free else None,
            "device_use_mib": ({"total": int(breakdown.group(3)), "model": int(breakdown.group(4)), "context": int(breakdown.group(5)),
                                "compute": int(breakdown.group(6))} if breakdown else None),
            "projected_vs_free_mib": [int(projected.group(1)), int(projected.group(2))] if projected else None}


def probe(model: Path, ctx: int, port: int = 18098) -> dict:
    """Start the CUDA llama-server once with -v until it listens, read the offload lines, stop it."""
    exe = next(CUDA_DIR.rglob("llama-server.exe"))
    log = ROOT / "artifacts/hardware" / f"probe-{model.stem}-{ctx}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen([str(exe), "-m", str(model), "-ngl", "99", "-c", str(ctx), "--host", "127.0.0.1",
                                    "--port", str(port), "--parallel", "1", "-v"], stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + 120
        while time.time() < deadline and process.poll() is None:
            if "listening on" in log.read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(0.5)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return offload(log.read_text(encoding="utf-8", errors="replace"))


def run_config(name, entry_name, device, ctx, extra_args):
    serving = load(ROOT / "labs/M09/llama_serving.py", "hw_llama_serving")
    local = serving.load("llama_local")
    if device == "gpu":
        if not any(CUDA_DIR.rglob("llama-server.exe")):
            return {"name": name, "status": "cuda_build_missing"}
        local.BIN = CUDA_DIR
    entry = serving.gguf_entry(entry_name)
    model = ROOT / entry["path"] / entry["file"]
    if not model.exists():
        return {"name": name, "status": "not_downloaded", "file": entry["file"]}
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if entry_name.startswith("qwen3.5") else None
    log = ROOT / "artifacts/hardware" / f"{name}.server.log"
    try:
        with local.Server(model, threads=CPU_THREADS, ctx=ctx, alias=entry_name, log=log, extra_args=extra_args) as server:
            pid = server.process.pid
            serving.chat_stream(local, "Hello", extra, entry_name)  # warm-up, not measured
            with serving.PeakRSS(pid) as ram, PeakVRAM(pid) as vram:
                answers = [serving.chat_stream(local, p, extra, entry_name) for p in serving.PROMPTS]
            load_seconds = server.load_seconds
    except (RuntimeError, TimeoutError) as error:
        return {"name": name, "status": "failed", "error": str(error),
                "offload": probe(model, ctx) if device == "gpu" else None}
    median = lambda key: round(statistics.median(a[key] for a in answers if a[key] is not None), 1)
    return {"name": name, "status": "measured", "model": entry_name, "file": entry["file"], "file_mib": round(model.stat().st_size / 2**20),
            "device": device, "ctx": ctx, "threads": CPU_THREADS, "server_args": extra_args, "load_seconds": round(load_seconds, 2),
            "ttft_ms_median": median("ttft_ms"), "prefill_tps_median": median("prefill_tps"),
            "decode_tps_median": median("decode_tps"), "generated_tokens": sum(a["generated_tokens"] or 0 for a in answers),
            "ram_peak_rss_mib": round(ram.peak / 2**20), "ram_peak_private_mib": round(ram.peak_private / 2**20),
            "vram_process_peak_mib": vram.process_mib or None, "vram_card_peak_mib": vram.card_mib or None,
            "offload": probe(model, ctx) if device == "gpu" else None, "sample_answer": answers[0]["text"][:200]}


def four_bit() -> dict:
    if not GPU_PYTHON.exists():
        return {"status": "gpu_env_missing", "hint": "create .venv-gpu (see quality/hardware/README.md)"}
    done = subprocess.run([str(GPU_PYTHON), "-I", str(ROOT / "tools/hardware_4bit.py")], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    if done.returncode != 0:
        return {"status": "failed", "stderr_tail": done.stderr[-2000:]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def machine() -> dict:
    import psutil

    gpu = nvidia("name,memory.total,driver_version")
    return {"os": f"{platform.system()} {platform.release()} ({platform.version()})", "cpu": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False), "cpu_logical": psutil.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "gpu": {"name": gpu[0][0], "vram_mib": int(gpu[0][1]), "driver": gpu[0][2]} if gpu else None}


def hypotheses(report: dict) -> list[dict]:
    """Verdicts on the plan's hypotheses (REFACTORING_PLAN_2.0, section 8), computed from the measured rows only."""
    rows = {r["name"]: r for r in report.get("llama_cpp", []) if r.get("status") == "measured"}
    out = []
    fits = {}
    for name in ("qwen3.5-4b-q4_k_m-gpu", "qwen3.5-4b-q4_k_m-gpu-ctx8k", "qwen3.5-4b-q4_k_m-gpu-ctx16k"):
        row = rows.get(name) or {}
        info = row.get("offload") or {}
        layers, projected = info.get("layers_on_gpu"), info.get("projected_vs_free_mib")
        if layers and projected:
            fits[row["ctx"]] = layers[0] == layers[1] and projected[0] <= projected[1]
    if fits:
        ok = [ctx for ctx, good in sorted(fits.items()) if good]
        out.append({"hypothesis": "Модель на 4 млрд параметров в Q4 целиком помещается в 4 ГБ видеопамяти с рабочим контекстом",
                    "verdict": "confirmed" if ok else "rejected",
                    "evidence": f"все слои на GPU; запас по памяти устройства при контексте {', '.join(map(str, ok)) or '—'}; "
                                f"не помещается без запаса: {', '.join(str(c) for c, g in sorted(fits.items()) if not g) or '—'} "
                                "(свободная память считается при открытом рабочем столе Windows)"})
    fourbit = report.get("torch_4bit") or {}
    if fourbit.get("status") in ("ok", "not_quantized", "failed"):
        out.append({"hypothesis": "4-битная загрузка (bitsandbytes) работает в Windows без WSL2",
                    "verdict": "confirmed" if fourbit["status"] == "ok" else "rejected",
                    "evidence": (f"bitsandbytes {fourbit.get('bitsandbytes')}, torch {fourbit.get('torch')}: "
                                 + ", ".join(f"{r['quantization']} — веса {r['weights_mib']} МиБ, пик CUDA {r['cuda_peak_allocated_mib']} МиБ"
                                             for r in fourbit.get("runs", [])))})
    out.append({"hypothesis": "QLoRA на ноутбуке возможна для моделей до 1,5 млрд параметров; для 3–4 млрд — на грани",
                "verdict": "not_tested",
                "evidence": "R06 проверяет только 4-битную загрузку и генерацию; обучение QLoRA проверяется в M12-L11X (R21)"})
    return out


def bench(only=None, probe_only=False):
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    rows = {r["name"]: r for r in report.get("llama_cpp", [])}
    for name, entry, device, ctx, extra in CONFIGS:
        if only and name not in only:
            continue
        if probe_only:
            if device == "gpu" and rows.get(name, {}).get("status") == "measured":
                model = ROOT / "models/gguf" / rows[name]["file"]
                rows[name]["offload"] = probe(model, ctx)
                print(name, json.dumps(rows[name]["offload"]), flush=True)
            continue
        print(f"measuring {name} ...", flush=True)
        rows[name] = run_config(name, entry, device, ctx, extra)
        print(json.dumps({k: v for k, v in rows[name].items() if k not in ("offload", "sample_answer")}, ensure_ascii=False), flush=True)
    report.update({"schema": 1, "task": "R06", "measured_on": date.today().isoformat(), "machine": machine(),
                   "llama_cpp_release": TAG, "llama_cpp": [rows[n] for n, *_ in CONFIGS if n in rows],
                   "method": "labs/M09/llama_serving.py prompts (6), greedy, max 64 new tokens, thinking off for Qwen3.5; "
                             "medians over prompts after one warm-up; prompt cache off"})
    if (not only or "4bit" in only) and not probe_only:
        print("4-bit PyTorch check ...", flush=True)
        report["torch_4bit"] = four_bit()
    report["hypotheses"] = hypotheses(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"report: {REPORT.relative_to(ROOT)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch-cuda", "bench"])
    parser.add_argument("--only", nargs="*", help="configuration names, or 4bit")
    parser.add_argument("--probe-only", action="store_true", help="only re-read the GPU offload of measured rows")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    fetch_cuda() if args.command == "fetch-cuda" else bench(args.only, args.probe_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
