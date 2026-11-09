"""A language model from scratch: byte-level BPE and a GPT of about 10M parameters on CPU (extension 1.2.0, T60, M08-L08X).

The corpus is the text of this course (lessons, keys, module pages, lab instructions; course-authored Russian Markdown,
about 2.7 MB). Whole files are split by a hash of their path: 85% train, 7.5% validation, 7.5% test. Everything here
was fixed before the first run:
- tokenizer: byte-level BPE with a GPT-2 style pre-tokenizer, 4096 tokens = 256 bytes + 3839 merges + <|endoftext|>;
  ties between equally frequent pairs go to the smaller pair of ids;
- models: "plan" d=352, 6 blocks, 8 heads (about 10M parameters) and "small" d=128, 4 blocks, 4 heads (about 1.4M),
  context 256, dropout 0.1, tied input and output embeddings, pre-LayerNorm, attention written out by hand;
- training: 1200 steps of 16 windows, AdamW (0.9, 0.95), weight decay 0.1 on matrices, peak learning rate 6e-4 with
  100 warmup steps and cosine decay to 6e-5, gradient clipping 1.0, seed 42; validation every 100 steps on 32 fixed
  windows, the checkpoint with the best validation loss is the one scored on the test files;
- metrics: test cross-entropy per token, perplexity, bits per byte (comparable across tokenizers), against a uniform,
  a unigram and a bigram (add-0.1) model of the same tokens, and against zlib, bz2 and lzma on the same test bytes
  (lzma also given the training text, so that it knows what the models know).

    python labs/M08/gpt_from_scratch.py run            # BPE and both models, about an hour on 8 CPU threads
    python labs/M08/gpt_from_scratch.py run --config small --steps 300   # a quick check, a few minutes
    python labs/M08/gpt_from_scratch.py report         # frozen run in quality/gpt-from-scratch, no training
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import heapq
import itertools
import json
import lzma
import math
import os
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M08/gpt"
FROZEN = ROOT / "quality/gpt-from-scratch"
SPECIAL = "<|endoftext|>"
PRETOKEN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
VOCAB_SIZE = 4096
CONFIGS = {"plan": {"dim": 352, "layers": 6, "heads": 8}, "small": {"dim": 128, "layers": 4, "heads": 4}}
TRAIN = {"context": 256, "batch": 16, "steps": 1200, "lr": 6e-4, "min_lr": 6e-5, "warmup": 100, "weight_decay": 0.1,
         "dropout": 0.1, "clip": 1.0, "seed": 42, "eval_every": 100, "eval_windows": 32}
PROMPTS = ["Языковая модель", "Индекс ускоряет", "Ответ кандидата"]


# ---------------------------------------------------------------- corpus

def corpus_files():
    files = sorted(ROOT.glob("course/lessons/*.md")) + sorted(ROOT.glob("course/modules/*.md")) + \
        sorted(ROOT.glob("solutions/modules/*.md")) + sorted(ROOT.glob("labs/*/README.md"))
    return [f for f in files if f.name != "M08-L08X.md"]  # the lesson about this lab does not train it


def split_of(path):
    """Whole files go to one split by a hash of their relative path: no paragraph is in two splits."""
    bucket = int(hashlib.sha256(path.relative_to(ROOT).as_posix().encode()).hexdigest(), 16) % 40
    return "test" if bucket < 3 else "validation" if bucket < 6 else "train"


def corpus():
    parts = {"train": [], "validation": [], "test": []}
    for path in corpus_files():
        parts[split_of(path)].append(path.read_text(encoding="utf-8"))
    return parts


# ---------------------------------------------------------------- BPE

class BPE:
    """Byte-level BPE: pieces of the pre-tokenizer are bytes, merges join the most frequent adjacent pair."""

    def __init__(self, merges=()):
        self.merges = [tuple(m) for m in merges]  # (left id, right id), new id = 256 + rank
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]
        self.special = 256 + len(self.merges)
        self.vocab[self.special] = SPECIAL.encode("utf-8")
        self.cache = {}

    @classmethod
    def train(cls, texts, vocab_size=VOCAB_SIZE):
        counts = Counter(piece for text in texts for piece in PRETOKEN.findall(text))
        words = [list(piece.encode("utf-8")) for piece in counts]
        freqs = list(counts.values())
        pairs, where = Counter(), defaultdict(set)
        for wi, word in enumerate(words):
            for pair in itertools.pairwise(word):
                pairs[pair] += freqs[wi]
                where[pair].add(wi)
        heap = [(-count, pair) for pair, count in pairs.items()]
        heapq.heapify(heap)
        merges = []
        while 256 + len(merges) < vocab_size - 1 and heap:
            count, pair = heapq.heappop(heap)
            if pairs.get(pair, 0) != -count or count == 0:
                continue  # a stale heap entry: the count changed after it was pushed
            new = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for wi in list(where[pair]):
                word, f = words[wi], freqs[wi]
                if not any(p == pair for p in itertools.pairwise(word)):
                    continue
                for p in itertools.pairwise(word):
                    pairs[p] -= f
                    changed.add(p)
                merged, i = [], 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                        merged.append(new)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[wi] = merged
                for p in itertools.pairwise(merged):
                    pairs[p] += f
                    where[p].add(wi)
                    changed.add(p)
            pairs.pop(pair, None)
            for p in changed:
                if pairs.get(p, 0) > 0:
                    heapq.heappush(heap, (-pairs[p], p))
        return cls(merges)

    def _piece(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        ids = list(piece.encode("utf-8"))
        while len(ids) > 1:
            ranked = [(self.ranks.get(p, math.inf), i) for i, p in enumerate(itertools.pairwise(ids))]
            rank, i = min(ranked)
            if rank == math.inf:
                break
            ids[i:i + 2] = [256 + rank]
        self.cache[piece] = ids
        return ids

    def encode(self, text):
        return [t for piece in PRETOKEN.findall(text) for t in self._piece(piece)]

    def encode_documents(self, documents):
        out = []
        for doc in documents:
            out += self.encode(doc) + [self.special]
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(json.dumps({"merges": self.merges, "special": SPECIAL}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["merges"])


# ---------------------------------------------------------------- model

def build_model(vocab, dim, layers, heads, context, dropout):
    import torch
    from torch import nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv, self.proj, self.drop = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim), nn.Dropout(dropout)
            self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

        def forward(self, x):
            b, t, _ = x.shape
            q, k, v = self.qkv(x).split(dim, dim=2)
            q, k, v = (z.view(b, t, heads, dim // heads).transpose(1, 2) for z in (q, k, v))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim // heads)  # (b, heads, t, t)
            scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))  # a position sees only itself and the past
            weights = self.drop(torch.softmax(scores, dim=-1))
            return self.proj((weights @ v).transpose(1, 2).reshape(b, t, dim))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.attn, self.ln2 = nn.LayerNorm(dim), Attention(), nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))  # pre-LayerNorm: the residual path stays an identity
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token, self.position = nn.Embedding(vocab, dim), nn.Embedding(context, dim)
            self.drop, self.blocks, self.ln = nn.Dropout(dropout), nn.ModuleList(Block() for _ in range(layers)), nn.LayerNorm(dim)
            self.context = context
            self.apply(self._init)
            for name, p in self.named_parameters():  # residual projections: smaller, so the sum of layers keeps its scale
                if name.endswith(("attn.proj.weight", "mlp.2.weight")):
                    nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

        @staticmethod
        def _init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        def forward(self, ids):
            t = ids.shape[1]
            x = self.drop(self.token(ids) + self.position(torch.arange(t)))
            for block in self.blocks:
                x = block(x)
            return self.ln(x) @ self.token.weight.T  # the output layer shares the token embedding

    return GPT()


def count_parameters(vocab, dim, layers, context):
    """Embeddings V*d + positions T*d, per block 12 d^2 + 13 d (attention 4d^2+4d, MLP 8d^2+5d, two LayerNorms 4d),
    final LayerNorm 2d; the output layer is tied and adds nothing."""
    return vocab * dim + context * dim + layers * (12 * dim * dim + 13 * dim) + 2 * dim


# ---------------------------------------------------------------- training

def windows(tokens, context, starts):
    import torch
    x = torch.stack([torch.tensor(tokens[s:s + context]) for s in starts])
    y = torch.stack([torch.tensor(tokens[s + 1:s + context + 1]) for s in starts])
    return x, y


def learning_rate(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def mean_loss(model, tokens, context, starts, batch=16):
    import torch
    import torch.nn.functional as F
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(starts), batch):
            x, y = windows(tokens, context, starts[i:i + batch])
            logits = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"))
            count += y.numel()
    return total / count


def train(name, tokens, vocab, cfg=TRAIN, threads=None):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(threads or min(8, os.cpu_count() or 1))
    shape = CONFIGS[name]
    model = build_model(vocab, shape["dim"], shape["layers"], shape["heads"], cfg["context"], cfg["dropout"])
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    keep = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": keep, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(cfg["seed"])
    span = cfg["context"] + 1
    val_starts = torch.linspace(0, len(tokens["validation"]) - span, cfg["eval_windows"]).long().tolist()
    history, best, started = [], (math.inf, None, -1), time.perf_counter()
    for step in range(cfg["steps"]):
        model.train()
        for group in opt.param_groups:
            group["lr"] = learning_rate(step, cfg)
        starts = torch.randint(0, len(tokens["train"]) - span, (cfg["batch"],), generator=generator).tolist()
        x, y = windows(tokens["train"], cfg["context"], starts)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        if step == 0 or (step + 1) % cfg["eval_every"] == 0:
            val = mean_loss(model, tokens["validation"], cfg["context"], val_starts)
            history.append({"step": step + 1, "train_loss": round(float(loss.detach()), 4), "val_loss": round(val, 4),
                            "seconds": round(time.perf_counter() - started, 1)})
            if val < best[0]:
                best = (val, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    return model, {"config": name, **shape, "parameters": sum(p.numel() for p in model.parameters()),
                   "best_step": best[2], "best_val_loss": round(best[0], 4), "history": history,
                   "train_seconds": round(seconds, 1), "tokens_per_second": round(cfg["steps"] * cfg["batch"] * cfg["context"] / seconds)}


def test_starts(tokens, context):
    return list(range(0, len(tokens) - context - 1, context))  # non-overlapping windows over the whole test split


def bits_per_byte(loss_per_token, tokens, nbytes):
    return loss_per_token * tokens / math.log(2) / nbytes


def ngram_baselines(train_ids, test_ids, vocab, alpha=0.1):
    """Uniform, unigram and add-alpha bigram models of the same tokens, scored per token on the test split."""
    unigram = Counter(train_ids)
    total = len(train_ids)
    uni = -sum(math.log((unigram[t] + alpha) / (total + alpha * vocab)) for t in test_ids[1:]) / (len(test_ids) - 1)
    bigram = Counter(itertools.pairwise(train_ids))
    bi = -sum(math.log((bigram[(a, b)] + alpha) / (unigram[a] + alpha * vocab)) for a, b in itertools.pairwise(test_ids)) / (len(test_ids) - 1)
    return {"uniform": math.log(vocab), "unigram": uni, "bigram": bi}


def compressor_baselines(train_text, test_text):
    """Bits per byte of the test text for general-purpose compressors. Alone they know nothing of the course; lzma
    "given train" is the extra size of compressing train + test over train alone, i.e. it has seen what the models saw."""
    train_bytes, test_bytes = train_text.encode("utf-8"), test_text.encode("utf-8")

    def xz(data):
        return len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))

    sizes = {"zlib": len(zlib.compress(test_bytes, 9)), "bz2": len(bz2.compress(test_bytes, 9)), "lzma": xz(test_bytes),
             "lzma_given_train": xz(train_bytes + test_bytes) - xz(train_bytes)}
    return {name: round(8 * size / len(test_bytes), 4) for name, size in sizes.items()}


def generate(model, bpe, prompt, steps=60, temperature=0.8, top_k=40, seed=7):
    import torch
    generator = torch.Generator().manual_seed(seed)
    ids = bpe.encode(prompt)
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            logits = model(torch.tensor([ids[-model.context:]]))[0, -1] / temperature
            top = torch.topk(logits, top_k)
            choice = top.indices[torch.multinomial(torch.softmax(top.values, -1), 1, generator=generator)]
            ids.append(int(choice))
            if ids[-1] == bpe.special:
                break
    return bpe.decode(ids)


def run(configs=("small", "plan"), steps=None, out_dir=OUT, threads=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**TRAIN, **({"steps": steps} if steps else {})}
    parts = corpus()
    started = time.perf_counter()
    bpe = BPE.train(parts["train"])
    bpe_seconds = time.perf_counter() - started
    bpe.save(out / "bpe.json")
    tokens = {k: bpe.encode_documents(v) for k, v in parts.items()}
    nbytes = {k: sum(len(d.encode("utf-8")) for d in v) for k, v in parts.items()}
    assert bpe.decode(tokens["test"]).replace(SPECIAL, "") == "".join(parts["test"])  # the tokenizer is lossless
    base = ngram_baselines(tokens["train"], tokens["test"], len(bpe.vocab))
    report = {"task": "T60", "corpus": {"files": {k: len(v) for k, v in parts.items()}, "bytes": nbytes, "tokens": {k: len(v) for k, v in tokens.items()}},
              "tokenizer": {"vocab": len(bpe.vocab), "merges": len(bpe.merges), "train_seconds": round(bpe_seconds, 1),
                            "bytes_per_token_test": round(nbytes["test"] / len(tokens["test"]), 3),
                            "first_merges": [bpe.vocab[256 + i].decode("utf-8", errors="replace") for i in range(20)],
                            "longest_tokens": sorted({bpe.vocab[i].decode("utf-8", errors="replace") for i in range(256, bpe.special)}, key=len)[-12:]},
              "baselines": {k: {"loss": round(v, 4), "bits_per_byte": round(bits_per_byte(v, len(tokens["test"]), nbytes["test"]), 4)} for k, v in base.items()},
              "compressors": compressor_baselines("".join(parts["train"]), "".join(parts["test"])),
              "train": cfg, "models": {}}
    for name in configs:
        model, info = train(name, tokens, len(bpe.vocab), cfg, threads)
        loss = mean_loss(model, tokens["test"], cfg["context"], test_starts(tokens["test"], cfg["context"]))
        info["test_loss"] = round(loss, 4)
        info["test_perplexity"] = round(math.exp(loss), 2)
        info["test_bits_per_byte"] = round(bits_per_byte(loss, len(tokens["test"]), nbytes["test"]), 4)
        info["samples"] = {p: generate(model, bpe, p) for p in PROMPTS}
        report["models"][name] = info
    report["collected"] = time.strftime("%Y-%m-%d")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def summary(report=None):
    report = report or json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    rows = [{"model": k, "loss": v["loss"], "bits_per_byte": v["bits_per_byte"]} for k, v in report["baselines"].items()]
    rows += [{"model": k, "loss": None, "bits_per_byte": v} for k, v in report.get("compressors", {}).items()]
    rows += [{"model": f"{k} ({v['parameters'] / 1e6:.1f}M)", "loss": v["test_loss"], "bits_per_byte": v["test_bits_per_byte"]}
             for k, v in report["models"].items()]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--config", choices=["small", "plan", "both"], default="both")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.command == "run":
        result = run(("small", "plan") if args.config == "both" else (args.config,), args.steps, threads=args.threads)
    else:
        result = None
    for row in summary(result):
        loss = "  -  " if row["loss"] is None else f"{row['loss']:.3f}"
        print(f"{row['model']:16} loss {loss}  bits/byte {row['bits_per_byte']:.3f}")
