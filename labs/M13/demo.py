"""Reproducible release, checked contract, measured calls and verified rollback."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def predict(release: dict, request: dict) -> dict:
    if set(request) != {"x"}:
        raise ValueError("exactly the field x is required")
    x = request["x"]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("x must be a finite number, not bool")
    if release["schema"] != 1:
        raise ValueError("unsupported release schema")
    return {"prediction": release["weight"] * x + release["bias"], "release": release["version"]}


def canary(release: dict) -> bool:
    cases = [(0, 1), (2, 5), (-2, -3)]
    return all(predict(release, {"x": x})["prediction"] == expected for x, expected in cases)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def run() -> dict:
    out = ROOT / "artifacts" / "M13"
    out.mkdir(parents=True, exist_ok=True)
    data = {"schema": 1, "rows": [[0, 1], [2, 5], [-2, -3]]}
    data_hash = hashlib.sha256(canonical(data)).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stable = {"schema": 1, "version": "1.0", "weight": 2.0, "bias": 1.0,
              "data_sha256": data_hash, "code_sha256": code_hash, "prompt": None}
    broken = dict(stable, version="1.1", weight=20.0)
    (out / "release-1.0.json").write_bytes(canonical(stable))
    (out / "release-1.1.json").write_bytes(canonical(broken))
    assert canary(stable)
    assert not canary(broken)
    active = stable
    previous = active
    active = broken
    detected = not canary(active)
    if detected:
        active = previous
    restored = canary(active)
    durations, logs = [], []
    for i in range(100):
        start = time.perf_counter()
        answer = predict(active, {"x": i / 10})
        elapsed = (time.perf_counter() - start) * 1000
        durations.append(elapsed)
        logs.append({"request_id": i, "release": answer["release"], "stage": "predict", "duration_ms": elapsed, "status": "ok"})
    rejected = 0
    for bad in ({"x": True}, {"x": float("nan")}, {"y": 2}, {"x": "2"}):
        try:
            predict(active, bad)
        except ValueError:
            rejected += 1
    assert rejected == 4 and restored and active["version"] == "1.0"
    (out / "requests.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n", encoding="utf-8")
    result = {"active_release": active["version"], "bad_release_detected": detected,
              "behavior_restored": restored, "invalid_requests_rejected": rejected,
              "n": len(durations), "p50_ms": statistics.median(durations),
              "p95_ms": percentile(durations, .95), "data_sha256": data_hash,
              "scope": "In-process CPU calls; these timings are not HTTP or production load measurements"}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
