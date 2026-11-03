"""Implement validation only. No optimizer, backward or parameter mutation is allowed."""
def validation_loss(model, x, y, criterion):
    """Return a Python float; restore the model's previous train/eval mode."""
    raise NotImplementedError('Your validation pass belongs here')
