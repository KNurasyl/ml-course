"""Return status, feedback, score, fallback_used. Never repair a score by guessing."""
def respond(raw):
    raise NotImplementedError('Parse exact feedback/score JSON or return a terminal needs_review response')
