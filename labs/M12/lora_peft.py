"""LoRA through PEFT on the M12-L07 task (extension 1.2.0, T44, lesson M12-L08X).

M12-L07 wrote the low-rank adapter by hand: rank 4, alpha 8, Q and V projections of the last four blocks of
SmolLM2-135M-Instruct, 24 fixed steps. Its saved result (quality/measurements-1.1.0/M12-sft-report.json) is negative:
check NLL fell from 3.53 to 0.33, yet only 2 of 4 check answers had the right command, and the old general questions
started to answer NEXT. This lab never rewrites that result; it reads it and compares:
- parity: the same adapter written as a PEFT LoraConfig gives the same logits as the hand-written module;
- same: PEFT LoRA with the M12-L07 initial A, data, order, steps, seed and learning rate, which should reproduce the run;
- wide: one hypothesis fixed before the run, that the adapter lacked capacity: rank 4 on every linear layer of all 30
  blocks, initialised the same way, everything else unchanged;
- artifact: save_pretrained with the base revision, reload into a fresh base, disable_adapter and merge_and_unload,
  each compared by logits.

    python labs/M12/lora_peft.py run       # needs peft and the local SmolLM2 of M09; a few minutes on CPU
    python labs/M12/lora_peft.py report    # frozen run in quality/lora, no model and no peft
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/lora"
OUT = ROOT / "artifacts/M12/lora"
SAVED_SFT = ROOT / "quality/measurements-1.1.0/M12-sft-report.json"
STEPS, LR, SEED, RANK, ALPHA, INIT_STD = 24, 0.006, 41, 4, 8, 0.02
# Fixed before the run. "same" must reproduce M12-L07; "wide" tests one hypothesis: too few adapted weights.
CONFIGS = {
    "same": {"target_modules": ["q_proj", "v_proj"], "last_layers": 4},
    "wide": {"target_modules": "all-linear", "last_layers": None},
}


def load(rel):
    spec = importlib.util.spec_from_file_location(f"m12_{Path(rel).stem}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lora_config(name, layer_count):
    from peft import LoraConfig
    spec = CONFIGS[name]
    layers = list(range(layer_count - spec["last_layers"], layer_count)) if spec["last_layers"] else None
    return LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=spec["target_modules"], layers_to_transform=layers, layers_pattern="layers" if layers else None)


def peft_name(manual_name):
    """model.layers.26.self_attn.q_proj.a -> base_model.model.model.layers.26.self_attn.q_proj.lora_A.default.weight"""
    module, matrix = manual_name.rsplit(".", 1)
    return f"base_model.model.{module}.lora_{'A' if matrix == 'a' else 'B'}.default.weight"


def base_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if "lora_" not in name:
            digest.update(p.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def init_like_m12(model, seed):
    """Every lora_A ~ N(0, 0.02) as in M12-L07, lora_B = 0: the start adds nothing, and B still gets a gradient."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if ".lora_A." in name:
                p.copy_(torch.randn(p.shape, generator=generator) * INIT_STD)
            elif ".lora_B." in name:
                p.zero_()


def train(model, rows):
    """The M12-L07 loop unchanged: AdamW without weight decay, one example per step in file order, clip 1.0."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    opt, history, start = torch.optim.AdamW(params, lr=LR, weight_decay=0.0), [], time.perf_counter()
    for step in range(STEPS):
        model.train()
        row = rows[step % len(rows)]
        opt.zero_grad(set_to_none=True)
        loss = model(**{k: v for k, v in row.items() if k != "prefix_length"}, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append(round(float(loss.detach()), 4))
    return history, time.perf_counter() - start


def stage(sft, tokenizer, model, data):
    train_rows, check_rows, old_rows = data
    result = {"train_nll": sft.mean_loss(model, train_rows), "check_nll": sft.mean_loss(model, check_rows),
              "old_nll": sft.mean_loss(model, old_rows), "check_generations": sft.generate_rows(tokenizer, model, sft.CHECK),
              "old_generations": sft.generate_rows(tokenizer, model, sft.OLD, "You are a concise assistant.")}
    result["task_review"] = sft.task_review(result["check_generations"])
    return result


def run(out_dir=OUT):
    import peft
    import torch
    import transformers
    from peft import PeftModel, get_peft_model
    sft, m09 = load("labs/M12/sft.py"), load("labs/M09/demo.py")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The M12-L07 random sequence: seed, load, attach. The hand-written A of that run is taken from here.
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer, manual, _ = m09.load_local()
    manual_names = sft.attach(manual)
    manual_a = {n: p.detach().clone() for n, p in manual.named_parameters() if n.endswith(".a")}
    data = ([sft.masked_example(tokenizer, q, a) for q, a in sft.TRAIN], [sft.masked_example(tokenizer, q, a) for q, a in sft.CHECK],
            [sft.masked_example(tokenizer, q, a, system="You are a concise assistant.") for q, a in sft.OLD])
    probe = data[1][0]["input_ids"]

    _, base, _ = m09.load_local()
    layer_count = len(base.model.layers)
    with torch.inference_mode():
        base_logits = base(probe).logits.clone()
    model = get_peft_model(base, lora_config("same", layer_count))
    state = dict(model.named_parameters())
    assert {peft_name(f"{n}.{m}") for n in manual_names for m in "ab"} == {n for n, p in state.items() if p.requires_grad}

    # Parity: copy the same A and a random nonzero B into both, compare logits, then return B to zero.
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in manual.named_parameters():
            if name.endswith(".b"):
                p.copy_(torch.randn(p.shape, generator=generator) * 0.01)
            if name.endswith((".a", ".b")):
                state[peft_name(name)].copy_(p)
        model.eval()
        manual.eval()
        parity_error = float((manual(probe).logits - model(probe).logits).abs().max())
        for name in manual_names:
            state[peft_name(name + ".a")].copy_(manual_a[name + ".a"])
            state[peft_name(name + ".b")].zero_()
    del manual

    runs = {}
    for name in CONFIGS:
        if name == "wide":
            torch.manual_seed(SEED)
            _, base, _ = m09.load_local()
            model = get_peft_model(base, lora_config("wide", layer_count))
            init_like_m12(model, SEED)
        frozen = base_hash(model)
        before = stage(sft, tokenizer, model, data)
        history, seconds = train(model, data[0])
        after = stage(sft, tokenizer, model, data)
        assert base_hash(model) == frozen, "the base weights changed"
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        runs[name] = {"target_modules": CONFIGS[name]["target_modules"], "last_layers": CONFIGS[name]["last_layers"],
                      "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in model.parameters()),
                      "adapted_modules": sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A")),
                      "training_seconds": round(seconds, 1), "history": history, "before": before, "after": after}
        if name == "same":
            # The artifact: PEFT format with the hub id and revision instead of a local path.
            model.peft_config["default"].base_model_name_or_path = m09.REPO
            model.peft_config["default"].revision = m09.REVISION
            model.save_pretrained(out / "same")
            model.eval()
            with torch.inference_mode():
                trained = model(probe).logits.clone()
                with model.disable_adapter():
                    disabled = model(probe).logits.clone()
            _, fresh, _ = m09.load_local()
            reloaded = PeftModel.from_pretrained(fresh, out / "same")
            reloaded.eval()
            with torch.inference_mode():
                reload_error = float((reloaded(probe).logits - trained).abs().max())
                merged = reloaded.merge_and_unload()
                merge_error = float((merged(probe).logits - trained).abs().max())
            files = {p.name: p.stat().st_size for p in sorted((out / "same").iterdir())}
            config = json.loads((out / "same/adapter_config.json").read_text(encoding="utf-8"))
            runs[name]["artifact"] = {"files": files, "reload_max_abs_error": reload_error, "merge_max_abs_error": merge_error,
                                      "disabled_vs_base_max_abs_error": float((disabled - base_logits).abs().max()),
                                      "adapter_vs_base_max_abs_difference": float((trained - base_logits).abs().max()),
                                      "config": {k: config[k] for k in ("base_model_name_or_path", "revision", "r", "lora_alpha",
                                                                        "target_modules", "layers_to_transform", "peft_type")}}
            del fresh, reloaded, merged
    report = {"task": "T44", "model": m09.REPO, "revision": m09.REVISION, "torch": torch.__version__,
              "transformers": transformers.__version__, "peft": peft.__version__, "seed": SEED, "steps": STEPS, "learning_rate": LR,
              "rank": RANK, "alpha": ALPHA, "init_std": INIT_STD, "parity_max_abs_error": parity_error, "runs": runs,
              "hypothesis_wide": "fixed before the run: too few adapted weights; all-linear in all blocks, everything else unchanged",
              "collected": time.strftime("%Y-%m-%d")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary(report)


def commands(stage_result):
    return [row["actual"].strip() for row in stage_result["check_generations"]], [row["actual"].strip() for row in stage_result["old_generations"]]


def summary(report=None):
    if report is None:
        report = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    saved = json.loads(SAVED_SFT.read_text(encoding="utf-8"))
    rows = [{"variant": "M12-L07 by hand (saved)", "trainable": saved["trainable_parameters"], "seconds": round(saved["training_seconds"], 1),
             "check_nll": [saved["before"]["check_nll"], saved["after"]["check_nll"]], "old_nll": [saved["before"]["old_nll"], saved["after"]["old_nll"]],
             "exact": saved["after"]["task_review"]["exact_match"], "wrong": len(saved["after"]["task_review"]["critical_wrong_command_indices"]),
             "check": commands(saved["after"])[0], "old": commands(saved["after"])[1]}]
    for name, r in report["runs"].items():
        rows.append({"variant": f"PEFT {name}", "trainable": r["trainable_parameters"], "seconds": r["training_seconds"],
                     "check_nll": [r["before"]["check_nll"], r["after"]["check_nll"]], "old_nll": [r["before"]["old_nll"], r["after"]["old_nll"]],
                     "exact": r["after"]["task_review"]["exact_match"], "wrong": len(r["after"]["task_review"]["critical_wrong_command_indices"]),
                     "check": commands(r["after"])[0], "old": commands(r["after"])[1]})
    same, base = report["runs"]["same"]["after"], saved["after"]
    reproduced = (max(abs(same[k] - base[k]) for k in ("train_nll", "check_nll", "old_nll")) < 1e-3
                  and commands(same) == commands(base))
    return {"task": "T44", "parity_max_abs_error": report["parity_max_abs_error"], "same_reproduces_m12_l07": reproduced,
            "artifact": report["runs"]["same"].get("artifact"), "rows": rows}


def table_text(result):
    lines = ["| Вариант | Обучаемых | Check NLL | Old NLL | Неверных команд |", "| --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        lines.append(f"| {r['variant']} | {r['trainable']} | {r['check_nll'][0]:.3f} → {r['check_nll'][1]:.3f} | "
                     f"{r['old_nll'][0]:.3f} → {r['old_nll'][1]:.3f} | {r['wrong']} из 4 |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    args = parser.parse_args()
    result = run() if args.command == "run" else summary()
    print(table_text(result))
    print(json.dumps({k: result[k] for k in ("parity_max_abs_error", "same_reproduces_m12_l07")}, ensure_ascii=False))
