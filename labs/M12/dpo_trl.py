"""DPO through TRL on top of the M12-L07 SFT (extension 1.2.0, T44, lesson M12-L09X).

The SFT policy is the M12-L07 adapter, retrained through PEFT exactly as in labs/M12/lora_peft.py (it reproduces the saved
run): it learned the format but answers NEXT to every check case, including the vague ones that need CLARIFY. DPO learns
from pairs instead of single targets: for each train prompt the right command is chosen and the other command rejected.
The SFT adapter is merged into the base and becomes the reference policy; TRL trains a new LoRA adapter on top, and with
the adapter disabled the same model is the reference. Settings are fixed before the run: beta 0.1, sigmoid loss,
24 steps of 2 pairs, constant learning rate 1e-3, seed 41.
Measured before and after: pair accuracy and implicit reward margin on the 4 held-out check pairs, check and old NLL,
free generations. The first logged loss must be ln 2: the new adapter starts at zero, so policy and reference coincide.

    python labs/M12/dpo_trl.py run       # needs peft, trl and the local SmolLM2 of M09; a few minutes on CPU
    python labs/M12/dpo_trl.py run --loss sigmoid sft --out artifacts/M12/dpo-sft   # task M12-L09X-A: add an SFT term
    python labs/M12/dpo_trl.py report    # frozen run in quality/dpo, no model and no trl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/dpo"
OUT = ROOT / "artifacts/M12/dpo"
BETA, STEPS, BATCH, LR, SEED = 0.1, 24, 2, 1e-3, 41
OLD_SYSTEM = "You are a concise assistant."


def load(rel):
    spec = importlib.util.spec_from_file_location(f"m12_{Path(rel).stem}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def other(command):
    return "NEXT: Explain one tradeoff." if command.startswith("CLARIFY") else "CLARIFY: Name your own action."


def pairs(cases, system):
    """Conversational preference pairs: the expected command is chosen, the other command rejected."""
    return [{"prompt": [{"role": "system", "content": system}, {"role": "user", "content": q}],
             "chosen": [{"role": "assistant", "content": a}], "rejected": [{"role": "assistant", "content": other(a)}]} for q, a in cases]


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=BETA):
    """-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))): the sigmoid loss of the DPO paper, per pair."""
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return math.log1p(math.exp(-margin))


def sequence_logp(model, row):
    """Sum of log-probabilities of the assistant tokens only, the quantity DPO compares."""
    import torch
    with torch.inference_mode():
        logits = model(input_ids=row["input_ids"], attention_mask=row["attention_mask"], use_cache=False).logits[:, :-1]
        labels = row["labels"][:, 1:]
        mask = labels != -100
        picked = torch.log_softmax(logits.float(), -1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return float((picked * mask).sum())


def first_token_prob(model, row):
    """Probability of the first answer token right after the prompt: what greedy decoding sees first."""
    import torch
    at = row["prefix_length"]
    with torch.inference_mode():
        logits = model(input_ids=row["input_ids"][:, :at], use_cache=False).logits[0, -1]
    return float(torch.softmax(logits.float(), -1)[row["input_ids"][0, at]])


def pair_scores(sft, tokenizer, model, cases):
    rows = []
    for q, a in cases:
        chosen_row, rejected_row = sft.masked_example(tokenizer, q, a), sft.masked_example(tokenizer, q, other(a))
        rows.append({"prompt": q, "chosen": a, "logp_chosen": round(sequence_logp(model, chosen_row), 3),
                     "logp_rejected": round(sequence_logp(model, rejected_row), 3),
                     "first_chosen": round(first_token_prob(model, chosen_row), 3), "first_rejected": round(first_token_prob(model, rejected_row), 3)})
    return rows


def evaluate(sft, lab, tokenizer, model, data, reference=None):
    result = lab.stage(sft, tokenizer, model, data)
    result["check_pairs"] = pair_scores(sft, tokenizer, model, sft.CHECK)
    result["pair_accuracy"] = sum(r["logp_chosen"] > r["logp_rejected"] for r in result["check_pairs"]) / len(sft.CHECK)
    if reference:  # implicit DPO reward: beta * (log pi - log pi_ref) on the held-out pairs
        margins = [BETA * ((p["logp_chosen"] - r["logp_chosen"]) - (p["logp_rejected"] - r["logp_rejected"]))
                   for p, r in zip(result["check_pairs"], reference)]
        result["check_reward_margins"] = [round(m, 3) for m in margins]
    return result


def run(out_dir=OUT, loss_types=("sigmoid",)):
    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import get_peft_model
    from trl import DPOConfig, DPOTrainer
    sft, m09, lab = load("labs/M12/sft.py"), load("labs/M09/demo.py"), load("labs/M12/lora_peft.py")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The SFT policy: the M12-L07 adapter retrained through PEFT, then merged into the base.
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer, base, _ = m09.load_local()
    model = get_peft_model(base, lab.lora_config("same", len(base.model.layers)))
    lab.init_like_m12(model, SEED)
    data = ([sft.masked_example(tokenizer, q, a) for q, a in sft.TRAIN], [sft.masked_example(tokenizer, q, a) for q, a in sft.CHECK],
            [sft.masked_example(tokenizer, q, a, system=OLD_SYSTEM) for q, a in sft.OLD])
    base_stage = evaluate(sft, lab, tokenizer, model, data)
    lab.train(model, data[0])
    policy = model.merge_and_unload()
    policy.eval()
    sft_stage = evaluate(sft, lab, tokenizer, policy, data)

    # 2. DPO with TRL: a new LoRA adapter on the merged SFT model; with the adapter disabled it is the reference.
    config = DPOConfig(output_dir=str(out / "trainer"), beta=BETA, loss_type=list(loss_types), learning_rate=LR, lr_scheduler_type="constant",
                       max_steps=STEPS, per_device_train_batch_size=BATCH, gradient_accumulation_steps=1, max_length=192,
                       logging_steps=1, save_strategy="no", report_to="none", use_cpu=True, seed=SEED, bf16=False, fp16=False,
                       dataloader_num_workers=0)
    trainer = DPOTrainer(model=policy, args=config, processing_class=tokenizer, train_dataset=Dataset.from_list(pairs(sft.TRAIN, sft.SYSTEM)),
                         peft_config=lab.lora_config("same", len(policy.model.layers)))
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    log = [{k: round(v, 4) for k, v in row.items() if isinstance(v, (int, float))} for row in trainer.state.log_history if "loss" in row]
    tuned = trainer.model
    tuned.eval()
    dpo_stage = evaluate(sft, lab, tokenizer, tuned, data, reference=sft_stage["check_pairs"])
    with tuned.disable_adapter():
        reference_stage = [{k: r[k] for k in ("logp_chosen", "logp_rejected")} for r in pair_scores(sft, tokenizer, tuned, sft.CHECK)]
    report = {"task": "T44", "model": m09.REPO, "revision": m09.REVISION, "torch": torch.__version__, "transformers": transformers.__version__,
              "peft": peft.__version__, "trl": trl.__version__, "beta": BETA, "loss_type": list(loss_types), "steps": STEPS, "batch": BATCH,
              "learning_rate": LR, "seed": SEED, "train_pairs": len(sft.TRAIN), "check_pairs": len(sft.CHECK),
              "trainable_parameters": sum(p.numel() for p in tuned.parameters() if p.requires_grad), "training_seconds": round(seconds, 1),
              "first_loss": log[0]["loss"], "ln2": round(math.log(2), 4),
              "reference_is_sft": reference_stage == [{k: r[k] for k in ("logp_chosen", "logp_rejected")} for r in sft_stage["check_pairs"]], "log": log,
              "stages": {"base": base_stage, "sft": sft_stage, "dpo": dpo_stage}, "collected": time.strftime("%Y-%m-%d")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary(report)


def summary(report=None):
    if report is None:
        report = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    rows = []
    for name, s in report["stages"].items():
        rows.append({"stage": name, "pair_accuracy": s["pair_accuracy"], "check_nll": s["check_nll"], "old_nll": s["old_nll"],
                     "exact": s["task_review"]["exact_match"], "wrong_commands": len(s["task_review"]["critical_wrong_command_indices"]),
                     "check": [r["actual"].strip() for r in s["check_generations"]], "old": [r["actual"].strip() for r in s["old_generations"]]})
    last = report["log"][-1]
    return {"task": "T44", "first_loss": report["first_loss"], "ln2": report["ln2"], "reference_is_sft": report["reference_is_sft"],
            "last_step": {k: last.get(k) for k in ("loss", "rewards/accuracies", "rewards/margins")}, "rows": rows}


def table_text(result):
    lines = ["| Стадия | Пары check | Check NLL | Old NLL | Неверных команд |", "| --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        lines.append(f"| {r['stage']} | {r['pair_accuracy']:.2f} | {r['check_nll']:.3f} | {r['old_nll']:.3f} | {r['wrong_commands']} из 4 |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--loss", nargs="+", default=["sigmoid"], help="TRL loss types, combined with equal weights")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.out, tuple(args.loss)) if args.command == "run" else summary()
    print(table_text(result))
    print(json.dumps({k: result[k] for k in ("first_loss", "ln2", "reference_is_sft", "last_step")}, ensure_ascii=False))
