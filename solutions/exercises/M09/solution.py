import json
def respond(raw):
    try:
        obj=json.loads(raw)
        good=isinstance(obj,dict) and set(obj)=={'feedback','score'} and type(obj['score']) is int and 0<=obj['score']<=2 and isinstance(obj['feedback'],str) and bool(obj['feedback'].strip())
    except (TypeError,ValueError):good=False
    if good:return {'status':'ok','feedback':obj['feedback'],'score':obj['score'],'fallback_used':False}
    return {'status':'needs_review','feedback':'Проверь ответ вручную по рубрике.','score':None,'fallback_used':True}
