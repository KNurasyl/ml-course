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


def embedders():
    """Optional 1.2.0 extension (M10-L07X): Russian-capable embedders, each over 1 GB, pinned to a revision."""
    from huggingface_hub import snapshot_download
    for config in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["embedders"]:
        directory = ROOT / config["path"]
        snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=directory,
            allow_patterns=config["allow_patterns"], ignore_patterns=["onnx/*", "*.pt", "*.onnx*"])
        print(f"Embedder ready: {directory}")


def rerankers():
    """Optional 1.2.0 extension (M10-L03, T35): cross-encoder reranker over 1 GB, pinned to a revision."""
    from huggingface_hub import snapshot_download
    for config in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["rerankers"]:
        directory = ROOT / config["path"]
        snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=directory,
            allow_patterns=config["allow_patterns"])
        print(f"Reranker ready: {directory}")



def guards():
    """Optional 1.2.0 extension (M14-L08X, T41): an English prompt-injection classifier, about 0.75 GB, pinned to a revision."""
    from huggingface_hub import snapshot_download
    for config in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["guards"]:
        directory = ROOT / config["path"]
        snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=directory,
            allow_patterns=config["allow_patterns"], ignore_patterns=config["ignore_patterns"])
        print(f"Guard classifier ready: {directory}")

def speech_extra():
    """Optional 1.2.0 extension (M15-L07X, T45): faster-whisper small (about 0.5 GB) and a CC0 Piper voice, pinned to revisions."""
    from huggingface_hub import hf_hub_download, snapshot_download
    for config in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["speech_extra"]:
        directory = ROOT / config["path"]
        if config["kind"] == "asr":
            snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=directory,
                              allow_patterns=config["allow_patterns"])
        else:
            for name in config["files"]:
                hf_hub_download(repo_id=config["repo_id"], filename=name, revision=config["revision"], local_dir=directory)
        print(f"Speech model ready: {directory}")


def gguf():
    """Optional 1.2.0 extension (M09-L07X, T30): GGUF files for llama.cpp, pinned to a revision and checked by sha256."""
    from huggingface_hub import hf_hub_download
    for config in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["gguf"]:
        target = ROOT / config["path"] / config["file"]
        if not target.exists():
            hf_hub_download(repo_id=config["repo_id"], filename=config["file"], revision=config["revision"],
                            local_dir=ROOT / config["path"])
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != config["sha256"]:
            raise SystemExit(f"sha256 mismatch for {target.name}")
        print(f"GGUF ready: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--embedders", action="store_true", help="optional M10-L07X models, about 4.9 GB in total")
    parser.add_argument("--rerankers", action="store_true", help="optional M10-L03 cross-encoder, about 2.3 GB")
    parser.add_argument("--gguf", action="store_true", help="optional M09-L07X GGUF files for llama.cpp, about 3.9 GB")
    parser.add_argument("--guards", action="store_true", help="optional M14-L08X prompt-injection classifier, about 0.75 GB")
    parser.add_argument("--speech-extra", action="store_true", help="optional M15-L07X faster-whisper small and a Piper voice, about 0.55 GB")
    args = parser.parse_args()
    if not (args.speech or args.llm or args.embedders or args.rerankers or args.gguf or args.guards or args.speech_extra):
        parser.error("Укажи --speech, --llm, --embedders, --rerankers, --gguf, --guards и/или --speech-extra")
    if args.speech:
        speech()
    if args.llm:
        language()
    if args.embedders:
        embedders()
    if args.rerankers:
        rerankers()
    if args.gguf:
        gguf()
    if args.guards:
        guards()
    if args.speech_extra:
        speech_extra()
