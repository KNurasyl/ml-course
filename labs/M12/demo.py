"""A real, deliberately tiny character bigram LM and frozen-base rank adapter."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def run(seed: int = 17) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    # Distinct complete sequences in train/test; the alphabet includes a boundary.
    # This synthetic task tests next-character patterns, not conversational ability.
    chars = "^abc$"
    index = {c: i for i, c in enumerate(chars)}

    def pairs(lines: list[str]):
        seqs = ["^" + line + "$" for line in lines]
        x = torch.tensor([index[a] for line in seqs for a in line[:-1]])
        y = torch.tensor([index[b] for line in seqs for b in line[1:]])
        return x, y

    train = pairs(["ab" * n for n in range(2, 10)])
    test = pairs(["ab" * n for n in range(10, 14)])
    changed_train = pairs(["ac" * n for n in range(2, 10)])
    changed_test = pairs(["ac" * n for n in range(10, 14)])
    base = nn.Embedding(len(chars), len(chars))
    with torch.no_grad():
        base.weight.zero_()
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(base.parameters(), lr=0.08)

    def score(logits, data):
        x, y = data
        with torch.no_grad():
            return float(loss(logits(x), y))

    initial = score(base, test)
    for _ in range(160):
        opt.zero_grad()
        value = loss(base(train[0]), train[1])
        value.backward()
        opt.step()
    before_old = score(base, test)
    before_new = score(base, changed_test)
    frozen = base.weight.detach().clone()
    base.requires_grad_(False)
    # Delta W = A @ B, rank <= 1; A random, B zero avoids zero gradients on both.
    a = nn.Parameter(torch.randn(len(chars), 1) * 0.05)
    b = nn.Parameter(torch.zeros(1, len(chars)))

    def adapted(x):
        return base(x) + a[x] @ b

    opt = torch.optim.Adam([a, b], lr=0.08)
    for _ in range(240):
        opt.zero_grad()
        value = loss(adapted(changed_train[0]), changed_train[1])
        value.backward()
        opt.step()
    after_new, after_old = score(adapted, changed_test), score(adapted, test)
    assert before_old < initial, "Base training must learn a next-character pattern"
    assert after_new < before_new, "Adaptation must improve held-out changed pattern"
    assert torch.equal(frozen, base.weight), "The base weights must remain frozen"
    result = {
        "seed": seed, "torch": torch.__version__, "device": "cpu",
        "initial_old_loss": initial, "base_old_loss": before_old,
        "base_new_loss": before_new, "adapted_new_loss": after_new,
        "adapted_old_loss": after_old, "base_parameters": base.weight.numel(),
        "adapter_parameters": a.numel() + b.numel(),
        "base_unchanged": torch.equal(frozen, base.weight),
        "limitation": "Synthetic bigram language model; no semantic or instruction-following ability",
    }
    out = ROOT / "artifacts" / "M12"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"alphabet": chars, "base": frozen, "a": a.detach(), "b": b.detach()}, out / "tiny-lm.pt")
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
