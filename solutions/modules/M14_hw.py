"""Reference lookup tool: strict args, server scope, budget, duplicate guard."""
import json

TERMS = {'s1': {'transaction': 'A unit of work committed or rolled back.',
                'index': 'A data structure supporting selected access paths.'},
         's2': {'private': 'Another session example.'}}


def execute(actions, session='s1', budget=4):
    observations, trace, seen = [], [], set()

    def result(stop, **extra):
        return dict(stop=stop, observations=observations, trace=trace, **extra)

    for step, action in enumerate(actions):
        if step >= budget:
            return result('budget')
        if not isinstance(action, dict) or set(action) != {'tool', 'args'}:
            return result('invalid_action')
        name, args = action['tool'], action['args']
        if not isinstance(args, dict):
            return result('denied')
        if name == 'finish':
            if set(args) != {'answer'} or not isinstance(args['answer'], str):
                return result('denied')
            return result('finished', answer=args['answer'])
        if name != 'lookup_term' or set(args) != {'term'} or not isinstance(args['term'], str):
            return result('denied')
        term = args['term'].strip().lower()
        if not 1 <= len(term) <= 80:
            return result('denied')
        key = (name, term)
        if key in seen:
            return result('repeated_call')
        seen.add(key)
        definition = TERMS.get(session, {}).get(term)
        if definition is None:
            return result('not_available')
        observations.append(definition)
        trace.append(dict(step=step, tool=name, term=term, status='ok'))
    return result('no_final_action')


def run():
    read = {'tool': 'lookup_term', 'args': {'term': 'transaction'}}
    final = {'tool': 'finish', 'args': {'answer': 'Transactions group operations.'}}
    cases = [
        ([read, final], 4, 'finished'),
        ([{'tool': 'shell', 'args': {}}], 4, 'denied'),
        ([{'tool': 'lookup_term', 'args': {'term': 1}}], 4, 'denied'),
        ([{'tool': 'lookup_term', 'args': {'term': 'index', 'session': 's2'}}], 4, 'denied'),
        ([{'tool': 'lookup_term', 'args': {'term': 'unknown'}}], 4, 'not_available'),
        ([{'tool': 'lookup_term', 'args': {'term': 'private'}}], 4, 'not_available'),
        ([read, {'tool': 'lookup_term', 'args': {'term': ' TRANSACTION '}}], 4, 'repeated_call'),
        ([read, final], 1, 'budget'),
        ([read], 4, 'no_final_action'),
        ([{'tool': 'finish', 'args': {'answer': 42}}], 4, 'denied'),
    ]
    outputs = [execute(actions, budget=budget) for actions, budget, _ in cases]
    assert [x['stop'] for x in outputs] == [expected for _, _, expected in cases]
    assert not outputs[5]['observations']
    return {'planner': 'scripted, not LLM', 'checks': 10, 'outcomes': [x['stop'] for x in outputs]}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
