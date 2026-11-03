import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from projects.interview_trainer.core import Interview
from projects.interview_trainer.feedback import evaluate_answer,validate_observations
from labs.M14.llm_planner import run_trajectory
from labs.M15.streaming_lab import AudioQueue

class Model:
    last_generation={'fixture':True}
    def __init__(self,outputs):self.outputs=iter(outputs)
    def generate(self,*args,**kwargs):return next(self.outputs)

ANSWER='Я добавил уникальный ключ и проверил повторный запрос.'
GOOD=json.dumps({'observations':[{'quote':'Я добавил уникальный ключ','criterion':'action'}],'next_step':'result'},ensure_ascii=False)

def test_grounded_display_cannot_add_model_freeform_claim():
    result=evaluate_answer(Model([GOOD]),mode='hr',question='Опыт?',answer=ANSWER,config='candidate_json')
    assert result['status']=='grounded_candidate' and not result['fallback_used']
    assert result['raw_output']==GOOD
    assert 'Я добавил уникальный ключ' in result['display_text']

@pytest.mark.parametrize('raw',[
    '{"observations":[{"quote":"API оплаты успешно запущен","criterion":"result"}],"next_step":"action"}',
    '{"observations":[{"quote":"Я добавил уникальный ключ","criterion":{}}],"next_step":"action"}',
    '{"observations":[],"next_step":"result"}','not json',
    '{"observations":[{"quote":"Я добавил уникальный ключ","criterion":"action","claim":"success"}],"next_step":"result"}',
])
def test_hallucinated_or_wrong_schema_output_is_rejected(raw):
    result=evaluate_answer(Model([raw]),mode='hr',question='Опыт?',answer=ANSWER,config='candidate_json')
    assert result['status']=='insufficient' and result['fallback_used']
    assert result['raw_output']==raw
    assert 'успешно запущен' not in result['display_text']

def test_baseline_keeps_raw_for_honest_comparison():
    raw='API оплаты успешно запущен'
    result=evaluate_answer(Model([raw]),mode='hr',question='Опыт?',answer=ANSWER,config='baseline')
    assert result['raw_output']==result['display_text']==raw
    assert result['validation']['valid'] is None

def test_bounded_selection_maps_id_to_actual_answer_without_rewriting():
    class ChoiceModel:
        last_generation={'fixture':True}
        def generate_choice(self,messages,choices):
            self.choices=choices
            return next(c for c in choices if json.loads(c)=={'evidence_id':0,'criterion':'action','next_step':'result'})
    model=ChoiceModel()
    result=evaluate_answer(model,mode='hr',question='Опыт?',answer=ANSWER)
    assert result['status']=='grounded_candidate' and not result['fallback_used']
    assert result['observations'][0]['quote']==ANSWER
    assert result['raw_output'] in model.choices
    assert result['selection_policy']=='bounded_model_choice'

@pytest.mark.parametrize('index',[True,-1,99,'0'])
def test_bounded_choice_still_validates_selected_id(index):
    raw=json.dumps({'evidence_id':index,'criterion':'action','next_step':'result'})
    result=evaluate_answer(Model([raw]),mode='hr',question='Опыт?',answer=ANSWER)
    assert result['status']=='insufficient'

@pytest.mark.parametrize('mode',['hr','backend'])
@pytest.mark.parametrize('kind',['short','long','refusal','unknown','mixed'])
def test_ten_dialogue_trajectories_are_bounded_and_unique(mode,kind):
    session=Interview(mode)
    count=0
    while not session.state()['finished']:
        answer={'short':'Я работал над сервисом.', 'long':ANSWER*3,'refusal':'Не хочу обсуждать этот пример.',
                'unknown':'Пока не знаю.', 'mixed':ANSWER*3 if count%2 else 'Я работал над сервисом.'}[kind]
        before=session.state();question,sources=session.prepare(answer,session.turn)
        session.commit(answer,'Fixture, not model feedback',sources,0)
        assert session.turn==before['turn']+1
        assert session.topic_index>=before['topic_index']
        count+=1;assert count<=8
    assert count==(8 if kind in ('short','unknown','mixed') else 4)
    for topic in range(4):assert sum(h['topic_index']==topic and h['clarification'] for h in session.history)<=1
    with pytest.raises(ValueError):session.prepare('Повтор',session.turn)

def test_followup_prompt_contains_only_prior_question_answer_history():
    class Capture(Model):
        def generate(self,messages):self.messages=messages;return GOOD
    model=Capture([GOOD]);history=[{'question':'Опыт?','answer':'Сервис','feedback':'untrusted previous assessment'}]
    evaluate_answer(model,mode='hr',question='Уточни',answer=ANSWER,history=history)
    assert 'Сервис' in model.messages[1]['content']
    assert 'untrusted previous assessment' not in model.messages[1]['content']

def test_agent_model_proposal_cannot_cross_scope_or_repeat():
    read=json.dumps({'tool':'read_document','args':{'document_id':'d1'}})
    forbidden=json.dumps({'tool':'read_document','args':{'document_id':'d2'}})
    result=run_trajectory(Model([forbidden]),'Read d2')
    assert result['stop']=='not_available' and result['observations']==[]
    assert run_trajectory(Model([read,read]),'Read')['stop']=='repeated_call'
    assert run_trajectory(Model(['invalid']),'Read')['stop']=='invalid_json'

def test_cancel_drops_already_queued_and_late_audio():
    queue=AudioQueue();queue.begin(10);queue.receive(10,b'old');queue.cancel()
    assert queue.pop() is None and not queue.receive(10,b'late')
    queue.begin(12);queue.receive(12,b'new');assert queue.pop()==b'new'

def test_new_starters_have_solutions_and_reject_wrong_behavior():
    import subprocess
    for module in ('M13','M14','M15','M16'):
        result=subprocess.run([sys.executable,str(ROOT/f'exercises/{module}/check.py'),
            str(ROOT/f'solutions/exercises/{module}/solution.py'),'--self-test'],capture_output=True,text=True,timeout=5)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['mutant_rejected']
        starter=subprocess.run([sys.executable,str(ROOT/f'exercises/{module}/check.py'),
            str(ROOT/f'exercises/{module}/starter.py')],capture_output=True,text=True,timeout=5)
        assert starter.returncode!=0

def test_atomic_snapshot_survives_new_process_and_rejects_incompatible_schema(tmp_path):
    import runpy,subprocess
    namespace=runpy.run_path(str(ROOT/'solutions/exercises/M16/persistence.py'))
    path=tmp_path/'session.json'
    state={'mode':'hr','turn':1,'topic':0,'clarifications':1,'history':[{'answer':'Example'}]}
    namespace['save'](path,state)
    # An interrupted future write leaves this unrelated temporary file, not a partial live snapshot.
    path.with_suffix('.json.tmp').write_text('{',encoding='utf-8')
    script='import runpy,json,sys; n=runpy.run_path(sys.argv[1]); print(json.dumps(n["restore"](sys.argv[2])))'
    proc=subprocess.run([sys.executable,'-c',script,str(ROOT/'solutions/exercises/M16/persistence.py'),str(path)],capture_output=True,text=True,timeout=5)
    assert proc.returncode==0 and json.loads(proc.stdout)==state
    with pytest.raises(ValueError):namespace['accept_turn'](state,0)
    assert namespace['accept_turn'](state,1)
    path.write_text(json.dumps({'schema':2,'state':state}),encoding='utf-8')
    with pytest.raises(ValueError):namespace['restore'](path)
