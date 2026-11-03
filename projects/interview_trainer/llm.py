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

    def generate_json(self, messages, max_new_tokens=192):
        return self.generate(messages, max_new_tokens=max_new_tokens, json_object=True)

    def generate_choice(self, messages, choices):
        if not choices or len(choices)>100 or not all(isinstance(c,str) and c for c in choices):
            raise ValueError('1..100 explicit string choices required')
        return self.generate(messages, max_new_tokens=256, choices=choices)

    def generate(self, messages, max_new_tokens=128, json_object=False, choices=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        import json
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
            # Stop generation after a complete emitted JSON object. This changes
            # the stop rule, never edits or substitutes the model's actual tokens.
            extra = {}
            if choices is not None:
                start=inputs.input_ids.shape[-1]
                encoded=[self.tokenizer.encode(choice,add_special_tokens=False) for choice in choices]
                eos=self.tokenizer.eos_token_id
                if eos is None: raise ValueError('EOS is required for bounded choices')
                def allowed(batch_id,input_ids):
                    prefix=input_ids[start:].tolist()
                    allowed_tokens=set()
                    for sequence in encoded:
                        if sequence[:len(prefix)]==prefix:
                            allowed_tokens.add(sequence[len(prefix)] if len(sequence)>len(prefix) else eos)
                    if not allowed_tokens:raise RuntimeError('generation left allowed prefix tree')
                    return sorted(allowed_tokens)
                extra['prefix_allowed_tokens_fn']=allowed
                extra['renormalize_logits']=True
                max_new_tokens=max(len(sequence) for sequence in encoded)+1
            if json_object:
                tokenizer, start = self.tokenizer, inputs.input_ids.shape[-1]
                class CompleteObject(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        text = tokenizer.decode(input_ids[0,start:], skip_special_tokens=True).strip()
                        try: value = json.loads(text)
                        except (ValueError,TypeError): return False
                        return isinstance(value,dict)
                extra['stopping_criteria']=StoppingCriteriaList([CompleteObject()])
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id, **extra)
            new_tokens = output[0, inputs.input_ids.shape[-1]:]
            stops = self.model.generation_config.eos_token_id
            stops = stops if isinstance(stops, list) else [stops]
            self.last_generation = {"input_tokens":int(inputs.input_ids.shape[-1]),"output_tokens":len(new_tokens),"stop_policy":"bounded_token_choices" if choices is not None else ("complete_json_or_limit" if json_object else "eos_or_limit"),
                "choice_count":len(choices) if choices is not None else None,
                "truncated":len(new_tokens) == max_new_tokens and int(new_tokens[-1]) not in stops}
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
