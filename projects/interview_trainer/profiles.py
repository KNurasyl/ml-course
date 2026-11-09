"""Model profiles of the interview trainer (IMPROVEMENT_PLAN.md 3.5, T50): read profiles.yaml, check, open a model.

    python -m projects.interview_trainer.profiles list
    python -m projects.interview_trainer.profiles check strong
    python -m projects.interview_trainer.profiles smoke strong     # one HR and one backend turn through the M11 rubric
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = Path(__file__).with_name("profiles.yaml")


class ProfileUnavailable(RuntimeError):
    """The profile cannot run here; the message says what is missing. Never replaced by another profile silently."""


def load_profiles(path: Path = PROFILES) -> dict:
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("default") not in data.get("profiles", {}):
        raise ValueError("profiles.yaml: default must name a profile")
    for name, profile in data["profiles"].items():
        if profile.get("backend") not in ("transformers", "llama_server", "openai_compatible"):
            raise ValueError(f"profiles.yaml: {name} has an unknown backend")
    return data


def get_profile(name: str | None = None, data: dict | None = None) -> tuple[str, dict]:
    data = data or load_profiles()
    name = name or data["default"]
    if name not in data["profiles"]:
        raise ValueError(f"unknown profile {name!r}; known: {', '.join(data['profiles'])}")
    return name, data["profiles"][name]


def _llama_local():
    spec = importlib.util.spec_from_file_location("trainer_llama_local", ROOT / "labs/M09/llama_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gguf_path(profile: dict) -> Path | None:
    if not profile.get("gguf"):
        return None
    registry = json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["gguf"]
    entry = next((e for e in registry if e["name"] == profile["gguf"]), None)
    if entry is None:
        raise ValueError(f"GGUF {profile['gguf']!r} is not in metadata/reference-models.json")
    return ROOT / entry["path"] / entry["file"]


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
        return False


def check(name: str | None = None, data: dict | None = None) -> list[str]:
    """What is missing for the profile; an empty list means it can run on this machine."""
    name, profile = get_profile(name, data)
    if profile.get("enabled", True) is False:
        return [profile.get("reason", "disabled in profiles.yaml")]
    missing = []
    if profile["backend"] == "transformers":
        if not (ROOT / profile["model_dir"] / "config.json").exists():
            missing.append(f"model not downloaded: {profile['download']}")
    elif profile["backend"] == "llama_server":
        path = gguf_path(profile)
        if path is None or not path.exists():
            missing.append(f"GGUF not downloaded: {profile.get('download', 'see profiles.yaml')}")
        if profile.get("start") and not healthy(profile["url"]) and not _llama_local().server_exe().exists():
            missing.append("llama-server binary not found: python labs/M09/llama_local.py fetch")
    else:
        if not profile.get("base_url") or not profile.get("model"):
            missing.append("base_url and model are not set in profiles.yaml")
        if not os.environ.get(profile.get("api_key_env", "")):
            missing.append(f"the key is not in the environment variable {profile.get('api_key_env')}")
    return missing


@contextlib.contextmanager
def open_model(name: str | None = None, data: dict | None = None):
    """Yield a model with the LocalLLM contract; a llama.cpp server started here is stopped on exit."""
    name, profile = get_profile(name, data)
    missing = check(name, data)
    if missing:
        raise ProfileUnavailable(f"profile {name}: " + "; ".join(missing))
    if profile["backend"] == "transformers":
        from .llm import LocalLLM
        yield LocalLLM(model_dir=ROOT / profile["model_dir"], threads=profile.get("threads", 4),
                       context_tokens=profile.get("context_tokens", 1800), profile=name)
        return
    from .llm_server import ServerLLM
    if profile["backend"] == "openai_compatible":
        yield ServerLLM(profile["base_url"], profile["model"], context_tokens=profile.get("context_tokens", 1800), llama_cpp=False,
                        api_key=os.environ[profile["api_key_env"]], profile=name)
        return
    extra = {"chat_template_kwargs": profile["chat_template_kwargs"]} if profile.get("chat_template_kwargs") else {}
    model = ServerLLM(profile["url"], profile["gguf"], context_tokens=profile.get("context_tokens", 1800), extra=extra, profile=name)
    if healthy(profile["url"]):
        yield model
        return
    port = urllib.parse.urlparse(profile["url"]).port
    log = ROOT / "artifacts/interview_trainer" / f"{name}.server.log"
    with _llama_local().Server(gguf_path(profile), threads=profile.get("threads", 4), ctx=profile.get("ctx", 4096),
                               alias=profile["gguf"], log=log, port=port) as server:
        job = bind_to_this_process(server.process)  # noqa: F841 - the open handle is what keeps the binding
        model.load_seconds = round(server.load_seconds, 1)
        yield model


def bind_to_this_process(process):
    """Windows: put the started server into a job object that is closed with this process. Ctrl+C stops the server
    through the context manager; the job also covers a closed console window or a killed app, which skip `finally`."""
    if os.name != "nt" or process is None:
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    handle = kernel32.OpenProcess(0x0100 | 0x0001, False, process.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    ok = bool(job and handle and kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limits
              and kernel32.AssignProcessToJobObject(job, handle))
    if handle:
        kernel32.CloseHandle(handle)
    if not ok:
        print(f"warning: llama-server is not bound to the app (error {ctypes.get_last_error()}); stop it with Stop-Process -Name llama-server")
    return job


SMOKE = [
    ("hr", "Расскажи о сложной задаче, которую ты довёл до результата.",
     "Я сам переписал медленный отчёт. Сначала измерил время, потом убрал лишние запросы. Отчёт стал строиться за 3 секунды вместо 40."),
    ("backend", "Зачем нужна транзакция в базе данных?",
     "Транзакция объединяет операции так, что применяются все или ни одной. Но длинная транзакция держит блокировки и мешает другим запросам."),
]


def smoke(name: str | None = None) -> dict:
    """One HR and one backend turn through feedback.evaluate_answer with the bounded-choice policy of the app."""
    from .feedback import evaluate_answer
    name, _ = get_profile(name)
    started = time.perf_counter()
    turns = []
    with open_model(name) as model:
        opened = time.perf_counter() - started
        for mode, question, answer in SMOKE:
            begin = time.perf_counter()
            result = evaluate_answer(model, mode=mode, question=question, answer=answer, config="candidate")
            turns.append({"mode": mode, "status": result["status"], "raw_output": result["raw_output"],
                          "observations": result["observations"], "seconds": round(time.perf_counter() - begin, 2), "generation": result["generation"]})
    return {"profile": name, "open_seconds": round(opened, 1), "load_seconds": getattr(model, "load_seconds", None), "turns": turns,
            "scope": "Two fixed answers: the profile runs end to end through the rubric; not a quality comparison (T51)."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "check", "smoke"])
    parser.add_argument("profile", nargs="?")
    args = parser.parse_args()
    if args.command == "list":
        data = load_profiles()
        for key in data["profiles"]:
            state = "ready" if not check(key, data) else "unavailable: " + "; ".join(check(key, data))
            print(f"{key:7} {'(default) ' if key == data['default'] else ''}{state}")
    elif args.command == "check":
        missing = check(args.profile)
        print(json.dumps({"profile": args.profile, "ready": not missing, "missing": missing}, ensure_ascii=False))
        raise SystemExit(1 if missing else 0)
    else:
        print(json.dumps(smoke(args.profile), ensure_ascii=False, indent=2))
