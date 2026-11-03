import torch
def validation_loss(model,x,y,criterion):
    was_training=model.training
    try:
        model.eval()
        with torch.inference_mode():return float(criterion(model(x),y))
    finally:model.train(was_training)
