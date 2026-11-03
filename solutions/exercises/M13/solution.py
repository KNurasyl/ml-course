def choose_release(previous, candidate, checks):
    required = {'schema', 'canary', 'regressions'}
    if set(checks) != required or any(value is not True for value in checks.values()):
        return previous
    if candidate.get('schema') != previous.get('schema'):
        return previous
    return candidate
