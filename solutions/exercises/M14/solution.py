import json
def parse_action(raw, allowed_ids):
    try: value=json.loads(raw)
    except (ValueError,TypeError): raise ValueError('invalid_json')
    if not isinstance(value,dict) or set(value)!={'tool','args'} or value['tool']!='read_document':
        raise ValueError('action_schema')
    args=value['args']
    if not isinstance(args,dict) or set(args)!={'document_id'} or not isinstance(args['document_id'],str):
        raise ValueError('args_schema')
    if args['document_id'] not in allowed_ids:raise ValueError('not_available')
    return value
