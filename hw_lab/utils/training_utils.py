import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_scores(model:nn.Module, criterion:nn.Module, dataloader:DataLoader):
    losses = []
    num_guessed = []
    
    model.eval()
    device = infer_device(model)

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x)
            losses.extend(criterion(preds, y).cpu().tolist())
            num_guessed.extend((preds.argmax(dim=1) == y).cpu().tolist())

    return np.mean(losses), np.mean(num_guessed)


def infer_device(module:nn.Module) -> torch.device:
    return next(module.parameters()).device

