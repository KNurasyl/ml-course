"""What a model card says against what the files say (route 2.0, lesson M09-L09X). Offline, CPU, no downloads.

For every local Hugging Face model folder: the YAML front matter of README.md (license, language, base_model,
pipeline_tag), the context and attention shape from config.json, the KV cache per token that shape implies and the
"tokenizer tax" — tokens per word on the same sentence in Russian and in English. For every local GGUF: architecture,
declared context length, layers and file size read from the GGUF header (labs/M09/llama_serving.py gguf_summary).

    python labs/M09/model_card.py            print the table and write artifacts/M09/model_card/report.json
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M09/model_card"
HF_MODELS = ["models/Qwen2.5-0.5B-Instruct", "models/smollm2-135m-instruct"]
# One sentence of the course company (data/company/kb/KB-001.md) and its translation.
RU = "Курьерская доставка по городу Алматы занимает один-два рабочих дня после подтверждения заказа."
EN = "Courier delivery within Almaty takes one to two business days after the order is confirmed."


def front_matter(readme: Path) -> dict:
    """The YAML block between the first two '---' lines of a model card."""
    import yaml

    text = readme.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    return yaml.safe_load(match.group(1)) if match else {}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_value: int = 2) -> int:
    """K and V for every layer with full attention: 2 x layers x kv_heads x head_dim x bytes (f16 = 2)."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def tokens_per_word(tokenizer, text: str) -> float:
    return len(tokenizer(text)["input_ids"]) / len(text.split())


def hf_model(folder: Path) -> dict:
    from transformers import AutoTokenizer

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    per_token = kv_bytes_per_token(config["num_hidden_layers"], kv_heads, head_dim)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    card = front_matter(folder / "README.md")
    return {"model": folder.name, "license": card.get("license"), "language": card.get("language"),
            "base_model": card.get("base_model"), "pipeline_tag": card.get("pipeline_tag"),
            "context": config.get("max_position_embeddings"), "layers": config["num_hidden_layers"],
            "kv_heads": kv_heads, "head_dim": head_dim, "kv_bytes_per_token": per_token,
            "kv_mib_at_context": round(per_token * config.get("max_position_embeddings", 0) / 2**20, 1),
            "ru_tokens_per_word": round(tokens_per_word(tokenizer, RU), 2),
            "en_tokens_per_word": round(tokens_per_word(tokenizer, EN), 2)}


def gguf_models() -> list[dict]:
    spec = importlib.util.spec_from_file_location("m09_serving_for_card", Path(__file__).with_name("llama_serving.py"))
    serving = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serving)
    rows = []
    for path in sorted((ROOT / "models/gguf").glob("*.gguf")):
        info = serving.gguf_summary(path)
        rows.append({"file": path.name, "architecture": info["architecture"], "context": info["context_length"],
                     "layers": info["layers"], "bits_per_weight": info["bits_per_weight"],
                     "file_mib": round(path.stat().st_size / 2**20), "has_chat_template": info["has_chat_template"]})
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    hf = [hf_model(ROOT / folder) for folder in HF_MODELS if (ROOT / folder / "config.json").is_file()]
    gguf = gguf_models() if (ROOT / "models/gguf").is_dir() else []
    for row in hf:
        print(f"{row['model']}: license={row['license']} language={row['language']} context={row['context']} "
              f"KV={row['kv_bytes_per_token']} B/token ({row['kv_mib_at_context']} MiB at full context) "
              f"tokens/word ru={row['ru_tokens_per_word']} en={row['en_tokens_per_word']}")
    for row in gguf:
        print(f"{row['file']}: {row['architecture']}, context {row['context']}, layers {row['layers']}, "
              f"{row['bits_per_weight']} bits/weight, {row['file_mib']} MiB")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({"hf": hf, "gguf": gguf, "sentence_ru": RU, "sentence_en": EN},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {(OUT / 'report.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
