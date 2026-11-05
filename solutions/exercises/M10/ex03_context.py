"""Reference for M10-ex03: fixed word windows with overlap and context selection by whole chunks."""


def window_chunks(n_words, size, overlap):
    for value, name in ((n_words, "n_words"), (size, "size"), (overlap, "overlap")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if n_words < 0 or size < 1 or not 0 <= overlap < size:
        raise ValueError("need n_words >= 0, size >= 1 and 0 <= overlap < size")
    step, windows, start = size - overlap, [], 0
    while start < n_words:
        windows.append((start, min(start + size, n_words)))
        if start + size >= n_words:
            break  # this window already reaches the end: another one would only repeat its tail
        start += step
    return windows


def select_context(ranked, active_versions, budget):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    ranked = list(ranked)
    ids = [chunk["id"] for chunk in ranked]
    if len(ids) != len(set(ids)):
        raise ValueError("ranked repeats a chunk ID")
    selected, excluded, texts, used = [], [], set(), 0
    for chunk in ranked:
        tokens = chunk["tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        document = chunk["document_id"]
        if document not in active_versions or active_versions[document] != chunk["version"]:
            excluded.append((chunk["id"], "obsolete"))  # the right topic in an old version is not the current answer
        elif chunk["text"] in texts:
            excluded.append((chunk["id"], "duplicate"))  # a repeat is not a second confirmation
        elif used + tokens > budget:
            excluded.append((chunk["id"], "budget"))  # skip it whole: cutting could drop a condition
        else:
            selected.append(chunk["id"])
            texts.add(chunk["text"])
            used += tokens
    return {"selected": selected, "excluded": excluded, "used": used}
