"""LoRA through PEFT and DPO through TRL (T44): the frozen runs keep the negative results, the saved M12-L07 result is
untouched, the PEFT adapter is the same arithmetic as the hand-written one, and the DPO loss starts at ln 2."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    spec = importlib.util.spec_from_file_location(f"t44_{Path(rel).stem}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lora, dpo, sft = load("labs/M12/lora_peft.py"), load("labs/M12/dpo_trl.py"), load("labs/M12/sft.py")
LORA = json.loads((ROOT / "quality/lora/report.json").read_text(encoding="utf-8"))
DPO = json.loads((ROOT / "quality/dpo/report.json").read_text(encoding="utf-8"))


def test_the_saved_m12_l07_result_is_untouched_and_still_negative():
    saved = json.loads(lora.SAVED_SFT.read_text(encoding="utf-8"))
    assert round(saved["before"]["check_nll"], 3) == 3.533 and round(saved["after"]["check_nll"], 3) == 0.332
    assert saved["release_allowed_by_loss_guard"] is True and saved["release_allowed"] is False
    assert saved["after"]["task_review"]["critical_wrong_command_indices"] == [0, 1]


def test_peft_reproduces_the_hand_written_adapter():
    result = lora.summary()
    assert result["parity_max_abs_error"] == 0.0 and result["same_reproduces_m12_l07"] is True
    same, saved = result["rows"][1], result["rows"][0]
    assert same["trainable"] == saved["trainable"] == 30720 and same["check"] == saved["check"] and same["old"] == saved["old"]


def test_the_peft_artifact_names_its_base_and_round_trips():
    art = LORA["runs"]["same"]["artifact"]
    assert art["config"]["base_model_name_or_path"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert art["config"]["revision"] == LORA["revision"] and art["config"]["layers_to_transform"] == [26, 27, 28, 29]
    assert art["reload_max_abs_error"] == 0.0 and art["disabled_vs_base_max_abs_error"] == 0.0
    assert art["merge_max_abs_error"] < 1e-3 < art["adapter_vs_base_max_abs_difference"]  # merging is exact up to float rounding
    assert art["files"]["adapter_model.safetensors"] < 200_000                             # about 125 KB next to a 540 MB base


def test_the_wide_hypothesis_failed_and_stays_failed():
    wide = LORA["runs"]["wide"]
    assert wide["trainable_parameters"] == 1_221_120 and wide["adapted_modules"] == 210
    assert wide["after"]["old_nll"] > 4 * wide["before"]["old_nll"]                       # old questions broke
    assert all(r["actual"].startswith("NEXT") for r in wide["after"]["check_generations"])
    # within each block of six same-class examples the loss falls near zero, then jumps when the class changes
    assert wide["history"][5] < 0.01 and wide["history"][6] > 5


def test_peft_names_follow_the_hand_written_ones():
    assert lora.peft_name("model.layers.26.self_attn.q_proj.a") == "base_model.model.model.layers.26.self_attn.q_proj.lora_A.default.weight"
    assert lora.peft_name("model.layers.29.self_attn.v_proj.b") == "base_model.model.model.layers.29.self_attn.v_proj.lora_B.default.weight"


def test_lora_config_is_the_m12_l07_adapter_in_peft_terms():
    pytest.importorskip("peft")
    import torch
    from peft import get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=5, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32)
    manual, base = LlamaForCausalLM(config), LlamaForCausalLM(config)
    base.load_state_dict(manual.state_dict())
    names = sft.attach(manual)  # the last four of five layers
    model = get_peft_model(base, lora.lora_config("same", 5))
    state = dict(model.named_parameters())
    with torch.no_grad():
        for name, p in manual.named_parameters():
            if name.endswith(".b"):
                p.normal_(std=0.1)
            if name.endswith((".a", ".b")):
                state[lora.peft_name(name)].copy_(p)
    ids = torch.randint(0, 64, (1, 12))
    manual.eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(manual(ids).logits, model(ids).logits, atol=1e-6)
    assert len(names) == 8 and sum(p.numel() for p in model.parameters() if p.requires_grad) == 4 * (4 * (32 + 32) + 4 * (32 + 16))  # q 32->32, v 32->16


def test_dpo_starts_at_ln2_and_its_margins_follow_from_the_log_probabilities():
    assert dpo.dpo_loss(-5.0, -3.0, -5.0, -3.0) == pytest.approx(math.log(2))  # policy equals reference
    assert DPO["first_loss"] == DPO["ln2"] == round(math.log(2), 4) and DPO["reference_is_sft"] is True
    sft_pairs, dpo_pairs = DPO["stages"]["sft"]["check_pairs"], DPO["stages"]["dpo"]["check_pairs"]
    margins = [dpo.BETA * ((d["logp_chosen"] - s["logp_chosen"]) - (d["logp_rejected"] - s["logp_rejected"])) for d, s in zip(dpo_pairs, sft_pairs)]
    assert [round(m, 3) for m in margins] == DPO["stages"]["dpo"]["check_reward_margins"]


def test_dpo_ranks_pairs_but_generation_keeps_the_wrong_command():
    stages = DPO["stages"]
    assert (stages["sft"]["pair_accuracy"], stages["dpo"]["pair_accuracy"]) == (0.5, 1.0)
    assert stages["dpo"]["task_review"]["critical_wrong_command_indices"] == [0, 1]
    for pair in stages["dpo"]["check_pairs"][:2]:  # vague answers: NEXT still comes first under greedy decoding
        assert pair["first_rejected"] > 0.5 > pair["first_chosen"]


def test_pairs_reject_the_other_command():
    row = dpo.pairs([("I helped the team.", "CLARIFY: Name your own action.")], "system")[0]
    assert row["chosen"][0]["content"].startswith("CLARIFY") and row["rejected"][0]["content"] == "NEXT: Explain one tradeoff."
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]


def test_the_sft_term_variant_fixes_the_commands_but_not_the_old_questions():
    variant = json.loads((ROOT / "quality/dpo/report-sigmoid-sft.json").read_text(encoding="utf-8"))
    assert variant["loss_type"] == ["sigmoid", "sft"] and variant["first_loss"] > DPO["first_loss"]  # ln 2 plus the NLL term
    after = variant["stages"]["dpo"]
    assert after["task_review"]["critical_wrong_command_indices"] == [] and after["task_review"]["exact_match"] == 1.0
    assert all(p["first_chosen"] > p["first_rejected"] for p in after["check_pairs"])
    assert after["old_nll"] > variant["stages"]["sft"]["old_nll"]
    assert not any(g["actual"].startswith("4") for g in after["old_generations"])


def test_reports_render_tables_without_a_model():
    assert lora.table_text(lora.summary()).count("\n") == 4 and dpo.table_text(dpo.summary()).count("\n") == 4
