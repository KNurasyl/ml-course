"""R06: the hardware report is parsed and judged by code; the session model is taken from it."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hardware_baseline", SOURCE / "tools/hardware_baseline.py")
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)

LOG = """0.00.297.310 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 3050 Ti Laptop GPU (4095 MiB, 3295 MiB free)
0.00.847.110 I load_tensors: offloaded 33/33 layers to GPU
0.00.882.097 I common_memory_breakdown_print: |   - CUDA0 (RTX 3050 Ti Laptop GPU) |  4095 = 3093 + (2991 =  2603 +     306 +      82) +       -2404 |
0.00.974.503 I common_params_fit_impl: projected to use 2991 MiB of device memory vs. 3243 MiB of free device memory
"""


def test_offload_lines_are_parsed():
    info = hw.offload(LOG)
    assert info["layers_on_gpu"] == [33, 33]
    assert info["device_free_before_load_mib"] == 3295
    assert info["device_use_mib"] == {"total": 2991, "model": 2603, "context": 306, "compute": 82}
    assert info["projected_vs_free_mib"] == [2991, 3243]
    assert hw.offload("nothing")["layers_on_gpu"] is None


def row(name, ctx, used, free=3243, layers=(33, 33)):
    return {"name": name, "status": "measured", "ctx": ctx,
            "offload": {"layers_on_gpu": list(layers), "projected_vs_free_mib": [used, free]}}


def test_hypotheses_follow_the_numbers():
    report = {"llama_cpp": [row("qwen3.5-4b-q4_k_m-gpu", 4096, 2859), row("qwen3.5-4b-q4_k_m-gpu-ctx16k", 16384, 3255)],
              "torch_4bit": {"status": "ok", "bitsandbytes": "x", "torch": "y",
                             "runs": [{"quantization": "nf4", "weights_mib": 430, "cuda_peak_allocated_mib": 520}]}}
    verdicts = {h["hypothesis"][:20]: h for h in hw.hypotheses(report)}
    fit = next(h for h in verdicts.values() if "4 ГБ" in h["hypothesis"])
    assert fit["verdict"] == "confirmed" and "4096" in fit["evidence"] and "16384" in fit["evidence"].split("без запаса:")[1]
    assert next(h for h in verdicts.values() if "bitsandbytes" in h["hypothesis"])["verdict"] == "confirmed"
    assert next(h for h in verdicts.values() if "QLoRA" in h["hypothesis"])["verdict"] == "not_tested"
    report["llama_cpp"] = [row("qwen3.5-4b-q4_k_m-gpu", 4096, 2859, layers=(20, 33))]
    report["torch_4bit"] = {"status": "failed"}
    assert {h["verdict"] for h in hw.hypotheses(report)} == {"rejected", "not_tested"}


def test_report_exists_and_session_model_points_to_it():
    report = json.loads((SOURCE / "quality/hardware/baseline.json").read_text(encoding="utf-8"))
    names = {r["name"] for r in report["llama_cpp"] if r["status"] == "measured"}
    assert {"qwen2.5-0.5b-q4_k_m-cpu", "qwen2.5-0.5b-q4_k_m-gpu", "qwen3.5-4b-q4_k_m-cpu", "qwen3.5-4b-q4_k_m-gpu"} <= names
    assert report["torch_4bit"]["status"] == "ok"
    route = yaml.safe_load((SOURCE / "course/route-v2.yaml").read_text(encoding="utf-8"))
    model = route["session_model"]
    assert model["profile"] in names and model["evidence"] == "quality/hardware/baseline.json"
    assert model["fallback_cpu"]["server_args"] == ["-ngl", "0"]  # invariant И7: a CPU path exists
