def training_example(prefix,answer,max_length,pad_id=0):
    if not prefix or not answer:raise ValueError('Need context and at least one assistant target')
    if type(max_length) is not int or max_length<len(prefix)+len(answer):raise ValueError('Overflow; do not truncate targets')
    n=max_length-len(prefix)-len(answer)
    return {'input_ids':list(prefix)+list(answer)+[pad_id]*n,
            'attention_mask':[1]*(len(prefix)+len(answer))+[0]*n,
            'labels':[-100]*len(prefix)+list(answer)+[-100]*n}
