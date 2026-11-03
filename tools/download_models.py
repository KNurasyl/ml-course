"""Explicit, resumable setup for the free local reference models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"


def speech():
    target = ROOT / "models/vosk-model-small-ru-0.22"
    archive = ROOT / "models/downloads/vosk-model-small-ru-0.22.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        if not archive.is_file():
            temporary = archive.with_suffix(".partial")
            urllib.request.urlretrieve(VOSK_URL, temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as package:
            base = (ROOT / "models").resolve()
            for member in package.infolist():
                if not (base / member.filename).resolve().is_relative_to(base):
                    raise ValueError("Unsafe ZIP member")
            package.extractall(base)
    result = {"name": target.name, "url": VOSK_URL, "path": str(target.relative_to(ROOT)),
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None}
    print(json.dumps(result, ensure_ascii=False))
    return result


def language():
    from huggingface_hub import snapshot_download
    config = json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["llm"]
    directory = ROOT / config["path"]
    snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE", "README.md", "*.jinja"])
    print(f"LLM ready: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    if not args.speech and not args.llm:
        parser.error("Укажи --speech и/или --llm")
    if args.speech:
        speech()
    if args.llm:
        language()
