from typing import Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.nn.utils import clip_grad_norm_

from utils.data_utils import load_dataset, build_dataloader, spoil_dataset
from utils.models import load_model
from utils.training_utils import compute_scores


class Trainer:
    def __init__(self, config:Dict):
        self.config = config
        self._init_dataloaders()
        self.model = load_model(self.config.get('model_type'), self.config.get('model_config', {})).to(self.config['device'])
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.get('lr', 1e-3))
        self._grad_clip_max_norm = config.get('grad_clip_max_norm', 1)
        
        self.train_history = {'losses': [], 'accs': []}
        self.test_history = {'losses': [], 'accs': [], 'iters': []}
        
        self.on_iter_done_callbacks = []        
        self.num_iters_done = 0
        self.max_num_iters = config.get('max_num_iters', 1000)
        self.val_freq_iters = config.get('val_freq_iters', -1)

        assert self.max_num_iters >= 0, 'When should I finish training?'
            
    def _init_dataloaders(self):
        self.train_ds = load_dataset(self.config['data_dir'], self.config.get('dataset_name', 'MNIST'), train=True)
        self.test_ds = load_dataset(self.config['data_dir'], self.config.get('dataset_name', 'MNIST'), train=False)
        
        num_good_points = self.config.get('num_good_points', len(self.train_ds))
        num_bad_points = self.config.get('num_bad_points', 0)

        self.train_ds = Subset(self.train_ds, range(num_good_points + num_bad_points))
        self.train_ds = spoil_dataset(self.train_ds, num_good_points, num_bad_points)
        
        self.train_dataloader = build_dataloader(self.train_ds, self.config['batch_size'], sequential=True)
        self.test_dataloader = build_dataloader(self.test_ds, self.config['batch_size'], sequential=True)
        
    @property
    def _bad_points_proportion(self):
        num_good_points = self.config.get('num_good_points', len(self.train_ds))
        num_bad_points = self.config.get('num_bad_points', 0)
        
        return num_bad_points / (num_bad_points + num_good_points)
        
    def _train_on_batch(self, batch):
        x, y = batch[0].to(self.config['device']), batch[1].to(self.config['device'])
        self.model.train()
        
        predictions = self.model(x)
        loss = self.criterion(predictions, y).mean()
        acc = (predictions.argmax(dim=1) == y).float().mean()
        
        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self._grad_clip_max_norm)
        self.optim.step()
        
        self.train_history['losses'].append(loss.item())
        self.train_history['accs'].append(acc.item())
        
    def _validate(self):
        loss, acc = compute_scores(self.model, self.criterion, self.test_dataloader)

        self.test_history['losses'].append(loss)
        self.test_history['accs'].append(acc)
        self.test_history['iters'].append(self.num_iters_done)
        
    def _on_iter_done(self):
        for fn in self.on_iter_done_callbacks:
            fn(self)
            
    def compute_train_accuracy(self):
        return compute_scores(self.model, self.criterion, self.train_dataloader)[1]
    
    def compute_test_accuracy(self):
        return compute_scores(self.model, self.criterion, self.test_dataloader)[1]

    def run_training(self, use_tqdm=False):
        iters = range(self.max_num_iters)
        iters = tqdm(iters) if use_tqdm else iters
        
        for _ in iters:
            indicies = np.random.choice(len(self.train_ds), self.config['batch_size'])
            batch = [self.train_ds[i] for i in indicies]
            batch = list(zip(*batch))
            batch = torch.stack(batch[0]), torch.Tensor(batch[1]).long()

            self._train_on_batch(batch)
            self.num_iters_done += 1
            self._on_iter_done()

            if self.val_freq_iters > 0 and self.num_iters_done % self.val_freq_iters == 0:
                self._validate()

            if self.num_iters_done > self.max_num_iters:
                break

        return self
