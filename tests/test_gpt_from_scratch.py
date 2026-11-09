"""A language model from scratch (T60): BPE merges checked by hand, lossless tokenisation, the GPT's parameter count,
causality, hand-written attention against PyTorch's, a tiny overfit, the schedule, n-gram baselines, the corpus split
and the frozen run; the full training is not repeated here."""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m08_gpt", ROOT / "labs/M08/gpt_from_scratch.py")
gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt)


def test_first_merges_follow_the_counts():
    # pieces: "low" x2 and " low" x2 after pre-tokenisation of "low low low low" ... counted by hand:
    # (l,o) and (o,w) appear 4 times each, the tie goes to the smaller pair of ids (108, 111) -> "lo";
    # then (lo, w) appears 4 times and wins over (" ", lo) with 3 -> "low"
    bpe = gpt.BPE.train(["low low low lower"], vocab_size=256 + 2 + 1)
    assert [bpe.vocab[256 + i] for i in range(2)] == [b"lo", b"low"]
    assert bpe.special == 258 and bpe.encode("low") == [257]


@pytest.mark.parametrize("text", ["Индекс ускоряет чтение.", "p95 = 200 мс 🚀\n\n  отступ", "naïve café ½", ""])
def test_tokenisation_is_lossless(text):
    bpe = gpt.BPE.train(["Индекс ускоряет чтение и запись. " * 20], vocab_size=300)
    assert bpe.decode(bpe.encode(text)) == text


def test_merges_survive_save_and_load(tmp_path):
    bpe = gpt.BPE.train(["транзакция транзакции транзакцию " * 10], vocab_size=280)
    bpe.save(tmp_path / "bpe.json")
    again = gpt.BPE.load(tmp_path / "bpe.json")
    assert again.encode("транзакция") == bpe.encode("транзакция") and again.special == bpe.special


def small_model(vocab=64, dim=32, layers=2, heads=4, context=16, dropout=0.0):
    torch.manual_seed(0)
    return gpt.build_model(vocab, dim, layers, heads, context, dropout)


def test_parameter_count_matches_the_formula():
    for vocab, dim, layers, context in ((64, 32, 2, 16), (4096, 352, 6, 256), (4096, 128, 4, 256)):
        model = gpt.build_model(vocab, dim, layers, 4 if dim % 4 == 0 else 8, context, 0.0)
        assert sum(p.numel() for p in model.parameters()) == gpt.count_parameters(vocab, dim, layers, context)
    assert gpt.count_parameters(4096, 352, 6, 256) == 10_481_152  # the "plan" model: about 10M


def test_a_position_never_sees_the_future():
    model = small_model().eval()
    ids = torch.randint(0, 64, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 64
    with torch.no_grad():
        a, b = model(ids), model(changed)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-6) and not torch.allclose(a[0, 8:], b[0, 8:])


def test_hand_written_attention_equals_pytorchs():
    model = small_model().eval()
    attn = model.blocks[0].attn
    x = torch.randn(2, 10, 32)
    q, k, v = attn.qkv(x).split(32, dim=2)
    q, k, v = (z.view(2, 10, 4, 8).transpose(1, 2) for z in (q, k, v))
    reference = attn.proj(torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 32))
    with torch.no_grad():
        assert torch.allclose(attn(x), reference, atol=1e-5)


def test_initial_loss_is_close_to_uniform():
    model = small_model(vocab=4096, dim=64, context=32).eval()
    ids = torch.randint(0, 4096, (4, 33))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).reshape(-1, 4096), ids[:, 1:].reshape(-1))
    assert abs(float(loss) - math.log(4096)) < 0.05  # small init: every token about 1/V


def test_a_tiny_model_overfits_one_sequence():
    torch.manual_seed(0)
    model = small_model(vocab=20, dim=32, layers=1, heads=2, context=16)
    seq = torch.tensor([[1, 5, 3, 9, 2, 7, 4, 8, 6, 1, 5, 3, 9, 2, 7, 4, 8]])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(80):
        loss = torch.nn.functional.cross_entropy(model(seq[:, :-1]).reshape(-1, 20), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.2  # the mechanics learn; generalisation is a separate question


def test_learning_rate_warms_up_and_decays():
    cfg = gpt.TRAIN
    assert gpt.learning_rate(0, cfg) == pytest.approx(cfg["lr"] / cfg["warmup"])
    assert gpt.learning_rate(cfg["warmup"], cfg) == pytest.approx(cfg["lr"])
    assert gpt.learning_rate(cfg["steps"], cfg) == pytest.approx(cfg["min_lr"])


def test_bigram_beats_unigram_on_structured_tokens():
    train, test = [0, 1, 2, 3] * 200, [0, 1, 2, 3] * 20
    base = gpt.ngram_baselines(train, test, vocab=8)
    assert base["bigram"] < base["unigram"] < base["uniform"] == pytest.approx(math.log(8))


def test_corpus_split_is_by_whole_files_and_skips_the_lesson():
    files = gpt.corpus_files()
    assert all(f.name != "M08-L08X.md" for f in files)
    splits = [gpt.split_of(f) for f in files]
    assert splits == [gpt.split_of(f) for f in files] and set(splits) == {"train", "validation", "test"}
    assert gpt.bits_per_byte(math.log(2), tokens=10, nbytes=10) == pytest.approx(1.0)


def test_lzma_given_the_training_text_compresses_it_away():
    rng = random.Random(0)
    text = "".join(rng.choice("абвгдежзик ") for _ in range(4000))
    sizes = gpt.compressor_baselines(text, text)
    assert set(sizes) == {"zlib", "bz2", "lzma", "lzma_given_train"}
    assert sizes["lzma_given_train"] < 0.2 * sizes["lzma"]  # the test text is already known: only a reference is left


# ---------------------------------------------------------------- the frozen run (quality/gpt-from-scratch)

FROZEN = ROOT / "quality/gpt-from-scratch"
REPORT = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))


def test_frozen_run_used_the_code_as_it_is():
    assert REPORT["train"] == gpt.TRAIN and REPORT["tokenizer"]["vocab"] == gpt.VOCAB_SIZE
    for name, model in REPORT["models"].items():
        shape = gpt.CONFIGS[name]
        assert model["parameters"] == gpt.count_parameters(gpt.VOCAB_SIZE, shape["dim"], shape["layers"], gpt.TRAIN["context"])
        assert model["best_val_loss"] == min(h["val_loss"] for h in model["history"])
        assert model["best_step"] == min(model["history"], key=lambda h: h["val_loss"])["step"]


def test_frozen_metrics_follow_their_formulas():
    tokens, nbytes = REPORT["corpus"]["tokens"]["test"], REPORT["corpus"]["bytes"]["test"]
    assert REPORT["tokenizer"]["bytes_per_token_test"] == pytest.approx(nbytes / tokens, abs=1e-3)
    for model in REPORT["models"].values():
        assert gpt.bits_per_byte(model["test_loss"], tokens, nbytes) == pytest.approx(model["test_bits_per_byte"], abs=1e-3)
        assert math.exp(model["test_loss"]) == pytest.approx(model["test_perplexity"], rel=1e-3)
    for base in REPORT["baselines"].values():
        assert gpt.bits_per_byte(base["loss"], tokens, nbytes) == pytest.approx(base["bits_per_byte"], abs=1e-3)


def test_frozen_run_supports_what_the_lesson_says():
    models, bpb = REPORT["models"], {k: v["bits_per_byte"] for k, v in REPORT["baselines"].items()} | REPORT["compressors"]
    # both models use context: below the bigram; the 10M model also beats lzma that has seen the training text
    assert models["plan"]["test_bits_per_byte"] < bpb["lzma_given_train"] < models["small"]["test_bits_per_byte"] < bpb["bigram"]
    assert bpb["bigram"] < min(bpb["zlib"], bpb["bz2"], bpb["lzma"])
    gaps = {}
    for name, model in models.items():
        assert model["best_step"] == REPORT["train"]["steps"]  # validation loss was still falling at the last step
        last = model["history"][-3:]
        gaps[name] = last[-1]["val_loss"] - sum(h["train_loss"] for h in last) / len(last)
    assert gaps["plan"] > 3 * gaps["small"] > 0  # but the larger model's train/validation gap is several times wider


def test_frozen_tokenizer_splits_a_sentence_as_the_lesson_shows():
    bpe = gpt.BPE.load(FROZEN / "bpe.json")
    assert len(bpe.vocab) == gpt.VOCAB_SIZE and bpe.special == gpt.VOCAB_SIZE - 1
    assert [bpe.decode([t]) for t in bpe.encode("Индекс ускоряет чтение.")] == ["Ин", "декс", " ускор", "яет", " чтение", "."]
    assert bpe.vocab[256] == b" " + bytes([0xD0])  # the first merge: a space and the lead byte of a Cyrillic letter
