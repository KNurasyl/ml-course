def playable(events):
    active=0;queue=[];played=[]
    for kind,turn,payload in events:
        if kind=='begin':
            if turn<=active:raise ValueError('turn must increase')
            active=turn;queue=[]
        elif kind=='receive':
            if turn==active:queue.append((turn,payload))
        elif kind=='cancel':
            active+=1;queue=[]
        elif kind=='play':
            if queue:
                owner,chunk=queue.pop(0)
                if owner==active:played.append(chunk)
        else:raise ValueError('unknown event')
    return played
