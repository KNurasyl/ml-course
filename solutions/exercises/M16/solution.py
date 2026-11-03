def transition(state, answer, limit):
    if type(limit) is not int or limit<0 or state['topic']>=state['topics']:raise ValueError('invalid state/limit')
    if not isinstance(answer,str) or len(answer.strip())<3:raise ValueError('invalid answer')
    out=dict(state);out['turn']+=1
    refusal=any(word in answer.lower() for word in ('не хочу','пропуст','отказываюсь'))
    incomplete=len(answer.strip())<80 or 'не знаю' in answer.lower()
    if incomplete and not refusal and state['clarifications']<limit:out['clarifications']+=1
    else:out['topic']+=1;out['clarifications']=0
    return out
