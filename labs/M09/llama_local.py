"""Prebuilt llama.cpp CPU binary inside the project and a local llama-server (extension 1.2.0, T30, lesson M09-L07X).

`fetch` downloads the pinned Windows CPU release from GitHub, checks its sha256 and unpacks it into runtime/llama.cpp
(ignored by git); nothing is installed system-wide. `Server` starts llama-server on 127.0.0.1 only, waits for
/health and stops it again. The server speaks the OpenAI-compatible HTTP API under /v1.

    python labs/M09/llama_local.py fetch
    python labs/M09/llama_local.py serve qwen2.5-0.5b-instruct-q4_k_m
    python labs/M09/llama_local.py serve --gpu      # session model of route 2.0 (R06): runtime, gguf, server_args
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = {"tag": "b11803", "published": "2026-10-21",
           "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11803/llama-b11803-bin-win-cpu-x64.zip",
           "sha256": "68a11f1f0ab78587bfc42071d2dbd4aa1b621357ddc375c1ee25a24e76d27954"}
BIN = ROOT / "runtime/llama.cpp" / RELEASE["tag"]
HOST, PORT = "127.0.0.1", 18080
ROUTE = "course/route-v2.yaml"


def fetch() -> Path:
    archive = BIN.parent / "download" / Path(RELEASE["url"]).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(RELEASE["url"], archive.with_suffix(".partial"))
        archive.with_suffix(".partial").replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != RELEASE["sha256"]:
        raise SystemExit(f"sha256 mismatch for {archive.name}: {digest}")
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if not (BIN / member.filename).resolve().is_relative_to(BIN.resolve()):
                raise ValueError("Unsafe ZIP member")
        package.extractall(BIN)
    print(f"llama.cpp {RELEASE['tag']} ready: {server_exe()}")
    return BIN


def server_exe(folder: Path | None = None) -> Path:
    cpu = folder is None  # BIN is read at call time, so tests can point it elsewhere
    folder = BIN if cpu else Path(folder)
    found = sorted(folder.rglob("llama-server.exe")) if folder.exists() else []
    if not found:
        if not cpu:  # the CUDA build of the session model comes from the hardware tool (R06)
            raise FileNotFoundError(f"No llama-server in {folder}. Run: .\\.venv\\Scripts\\python.exe tools\\hardware_baseline.py fetch-cuda")
        raise FileNotFoundError("Run: .\\.venv\\Scripts\\python.exe labs\\M09\\llama_local.py fetch")
    return found[0]


def gguf_path(name: str, root: Path = ROOT) -> Path:
    entry = next(e for e in json.loads((root / "metadata/reference-models.json").read_text(encoding="utf-8"))["gguf"] if e["name"] == name)
    return root / entry["path"] / entry["file"]


def session_profile(root: Path = ROOT) -> dict:
    """runtime, gguf and server_args of session_model in course/route-v2.yaml, turned into Server arguments."""
    import yaml

    model = yaml.safe_load((root / ROUTE).read_text(encoding="utf-8"))["session_model"]
    args, ctx, extra = [str(a) for a in model["server_args"]], None, []
    while args:
        flag = args.pop(0)
        if flag in ("-c", "--ctx-size"):
            ctx = int(args.pop(0))
        elif flag in ("-np", "--parallel"):
            if args.pop(0) != "1":  # Server always starts one slot
                raise ValueError("session_model: only --parallel 1 is supported here")
        else:
            extra.append(flag)
    return {"name": model["gguf"], "model": gguf_path(model["gguf"], root), "runtime": root / model["runtime"],
            "ctx": ctx or 4096, "extra_args": extra}


def request(path, payload=None, timeout=600, port=PORT):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{HOST}:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


class Server:
    """One llama-server process for one GGUF model; logs go to a file, never to pipes the parent would wait on."""

    def __init__(self, model: Path, threads=4, ctx=4096, alias: str | None = None, log: Path | None = None, port=PORT, extra_args=(),
                 runtime: Path | None = None):
        self.runtime = runtime  # None: the CPU build in BIN; the GPU session model has its own CUDA build
        self.model, self.threads, self.ctx, self.alias, self.port = Path(model), threads, ctx, alias or Path(model).stem, port
        self.extra_args = [str(a) for a in extra_args]  # e.g. ["--cache-ram", "0"] for the trainer's strong profile (T51)
        self.log = Path(log) if log else ROOT / "artifacts/M09/llama" / f"{self.model.stem}.server.log"
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        try:  # a server left from an interrupted run would answer instead of the new one
            with request("/health", timeout=2, port=self.port):
                raise RuntimeError(f"port {self.port} is busy: stop the previous llama-server (Stop-Process -Name llama-server)")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w", encoding="utf-8")
        command = [str(server_exe(self.runtime)), "-m", str(self.model), "--host", HOST, "--port", str(self.port), "-t", str(self.threads),
                   "-c", str(self.ctx), "--parallel", "1", "--jinja", "--no-webui", "--alias", self.alias, *self.extra_args]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.started = time.perf_counter()
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, creationflags=flags)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}, see {self.log}")
            try:
                with request("/health", timeout=5, port=self.port) as response:
                    if response.status == 200:
                        self.load_seconds = time.perf_counter() - self.started
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.25)
        raise TimeoutError(f"llama-server did not become healthy, see {self.log}")

    def __exit__(self, *exc):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


def serve(name, threads=4, ctx=4096, port=PORT, gpu=False):
    """Keep one server running in the terminal until Ctrl+C, for requests by hand from PowerShell, Node or Python.

    With gpu=True the model, runtime and arguments come from session_model of route 2.0, not from name and ctx."""
    if gpu:
        profile = session_profile()
        name = profile["name"]
        server = Server(profile["model"], threads=threads, ctx=profile["ctx"], alias=name, port=port,
                        extra_args=profile["extra_args"], runtime=profile["runtime"])
    else:
        server = Server(gguf_path(name), threads=threads, ctx=ctx, alias=name, port=port)
    with server:
        print(f"{name}: http://{HOST}:{port}/v1 (log: {server.log}); Ctrl+C stops the server")
        try:
            server.process.wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "serve"])
    parser.add_argument("model", nargs="?", default="qwen2.5-0.5b-instruct-q4_k_m", help="a gguf name from metadata/reference-models.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=PORT, help="a second model needs its own port, e.g. 18081")
    parser.add_argument("--gpu", action="store_true",
                        help="serve session_model of course/route-v2.yaml: its runtime, gguf and server_args (model and --ctx are ignored)")
    args = parser.parse_args()
    fetch() if args.command == "fetch" else serve(args.model, args.threads, args.ctx, args.port, gpu=args.gpu)
