"""Pinned small real LLM, measured locally. No API or remote executable model code."""
from pathlib import Path
import argparse, json, os, time
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[2]
MODEL_DIR=ROOT/'models/smollm2-135m-instruct'
REPO='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'

def download():
    from huggingface_hub import snapshot_download
    files=['model.safetensors','config.json','generation_config.json','tokenizer.json',
           'tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','README.md']
    snapshot_download(REPO,revision=REVISION,local_dir=str(MODEL_DIR),allow_patterns=files)
    (MODEL_DIR/'course-provenance.json').write_text(json.dumps({'repo_id':REPO,'revision':REVISION}),encoding='utf-8')

def load_local():
    if not (MODEL_DIR/'model.safetensors').exists():
        raise FileNotFoundError('Run: .\\.venv\\Scripts\\python.exe labs\\M09\\demo.py --download')
    torch.set_num_threads(2)
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,dtype=torch.float32,
                                             attn_implementation='eager')
    model.eval()
    return tokenizer,model,time.perf_counter()-start

def generate(tokenizer,model,messages,max_new_tokens=24):
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                                        return_dict=True,return_tensors='pt')
    prompt_tokens=int(inputs['input_ids'].shape[1])
    context=int(model.config.max_position_embeddings)
    if prompt_tokens+max_new_tokens>context:raise ValueError('context budget exceeded')
    start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    elapsed=time.perf_counter()-start
    new=output[0,prompt_tokens:]
    return {'text':tokenizer.decode(new,skip_special_tokens=True),'prompt_tokens':prompt_tokens,
            'generated_tokens':len(new),'seconds':elapsed,'tokens_per_second':len(new)/elapsed,
            'context_limit':context,'max_new_tokens':max_new_tokens}

def validate_feedback(text):
    try:value=json.loads(text)
    except (json.JSONDecodeError,TypeError):return {'valid':False,'reason':'invalid_json'}
    if not isinstance(value,dict) or set(value)!={'feedback','score'}:
        return {'valid':False,'reason':'wrong_keys'}
    if not isinstance(value['feedback'],str) or not value['feedback'].strip():
        return {'valid':False,'reason':'empty_feedback'}
    if type(value['score']) is not int or not 0<=value['score']<=2:
        return {'valid':False,'reason':'score_out_of_range'}
    return {'valid':True,'value':value}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    if args.download:download()
    tokenizer,model,load_seconds=load_local()
    messages=[{'role':'system','content':'You are a concise assistant.'},{'role':'user','content':'What is a SQL transaction? Answer in one sentence.'}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    result=generate(tokenizer,model,messages)
    structured=generate(tokenizer,model,[{'role':'user','content':'Return JSON only with keys feedback and score. feedback must be a short string; score must be integer 0, 1 or 2. Evaluate: A SQL transaction groups operations.'}],max_new_tokens=48)
    validation=validate_feedback(structured['text'])
    assert validate_feedback('{"feedback":"Clear","score":1}')['valid']
    assert not validate_feedback('{"feedback":"","score":1}')['valid']
    assert not validate_feedback('{"feedback":"x","score":true}')['valid']
    assert not validate_feedback('hello')['valid']
    assert result['generated_tokens']>0 and result['prompt_tokens']+result['generated_tokens']<=result['context_limit']
    report={'repo_id':REPO,'revision':REVISION,'device':'cpu','dtype':'float32','torch':torch.__version__,
            'load_seconds':load_seconds,'parameter_count':sum(p.numel() for p in model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'rendered_prompt':rendered,'generation':result,'structured_generation':structured,
            'validation':validation,'fallback_used':not validation['valid'],
            'scope':'English small model; no claim of Russian interview quality or guaranteed valid JSON'}
    out=ROOT/'artifacts/M09';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rendered_prompt'},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
