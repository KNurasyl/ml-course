def search(ids, documents, query, k):
    """Return [(id, cosine), ...], decreasing score then id; reject invalid shapes,
    zero/nonfinite vectors, duplicate IDs, k outside 1..len(ids)."""
    raise NotImplementedError("Implement search")

def recall(found, relevant):
    """Set-based recall; raise ValueError when relevant is empty."""
    raise NotImplementedError("Implement recall")
