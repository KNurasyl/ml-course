def retrieval_scores(ranking,relevant,k,abstained):
    if type(k) is not int or k<1 or len(ranking)!=len(set(ranking)):raise ValueError('invalid k or duplicate ranking')
    gold=set(relevant)
    if not gold:return {'answerable':False,'recall':None,'mrr':None,'correct_abstention':bool(abstained)}
    rank=next((i for i,v in enumerate(ranking,1) if v in gold),None)
    return {'answerable':True,'recall':len(set(ranking[:k])&gold)/len(gold),'mrr':0. if rank is None else 1/rank,'correct_abstention':None}
