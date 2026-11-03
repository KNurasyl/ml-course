"""Real local CPU inference; model failure is visible, never replaced with canned feedback."""
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[2]


class LocalLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.lock = threading.Lock()
        self.last_generation = {}

    def generate(self, messages, max_new_tokens=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        with self.lock:
            if self.model is None:
                directory = ROOT / "models/Qwen2.5-0.5B-Instruct"
                if not (directory / "config.json").exists():
                    raise FileNotFoundError("Выполни python tools/download_models.py --llm")
                torch.set_num_threads(4)
                self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                    dtype=torch.float32).eval()
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(rendered, return_tensors="pt")
            if inputs.input_ids.shape[-1] > 1800:
                raise ValueError("Контекст превышает учебный лимит 1800 токенов: сократи ответ")
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
            new_tokens = output[0, inputs.input_ids.shape[-1]:]
            stops = self.model.generation_config.eos_token_id
            stops = stops if isinstance(stops, list) else [stops]
            self.last_generation = {"input_tokens":int(inputs.input_ids.shape[-1]),"output_tokens":len(new_tokens),
                "truncated":len(new_tokens) == max_new_tokens and int(new_tokens[-1]) not in stops}
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
