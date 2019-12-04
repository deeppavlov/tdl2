from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn


DEFAULT_WIDTH = 1000
    

class DenseModel(nn.Module):
    def __init__(self, config:Dict):
        super(DenseModel, self).__init__()

        width = config.get('width', DEFAULT_WIDTH)
        
        self.nn = nn.Sequential(
            Flatten(),
            nn.Linear(784, width, bias=False),
            nn.ReLU(),
            nn.Linear(width, 10, bias=False),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


def load_model(model_type:str, hp:Dict=None) -> nn.Module:
    hp = hp or {}
    
    if model_type == 'DenseModel':
        return DenseModel(hp)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')