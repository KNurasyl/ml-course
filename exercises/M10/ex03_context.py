def window_chunks(n_words, size, overlap):
    """Fixed word windows as half-open (start, end) pairs: step = size - overlap, starts 0, step, 2 * step, ...

    Stop at the first window that reaches the end; its end is n_words. n_words = 0 gives [].
    Reject n_words < 0, size < 1, overlap < 0, overlap >= size and non-int or bool arguments.
    """
    raise NotImplementedError("Implement window_chunks")


def select_context(ranked, active_versions, budget):
    """Pick whole chunks in rank order within a token budget; never cut a chunk.

    ranked: list of dicts with "id", "document_id", "version", "tokens" (positive int) and "text".
    active_versions: {document_id: active version}. Check each chunk in this order and skip it as
    "obsolete" when its document has no active version or its version is not the active one,
    as "duplicate" when its text equals the text of an already selected chunk, and as "budget" when
    its tokens do not fit what is left; a later, smaller chunk may still fit.
    Return {"selected": [ids], "excluded": [(id, reason), ...], "used": tokens of the selected chunks}.
    Reject a negative, non-int or bool budget, tokens that are not a positive int and repeated IDs.
    """
    raise NotImplementedError("Implement select_context")
