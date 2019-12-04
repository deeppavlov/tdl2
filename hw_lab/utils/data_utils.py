import os
import random

import numpy as np
from torch.utils.data import ConcatDataset, Subset, Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor


def shuffle_labels(dataset:Dataset, shuffle_ratio:float=0) -> Dataset:
    """Takes dataset and shuffles its labels"""
    if shuffle_ratio == 0:
        return dataset
    
    inputs, targets = zip(*dataset)
    num_inputs_to_shuffle = int(len(dataset) * shuffle_ratio)
    shuffled_inputs_idx = random.sample(range(len(dataset)), k=num_inputs_to_shuffle)
    
    # Shuffling
    targets = np.array(targets)
    targets[shuffled_inputs_idx] = np.random.permutation(targets[shuffled_inputs_idx])

    shuffled_dataset = [(x,y) for x,y in zip(inputs, targets)]
    
    return shuffled_dataset


def load_dataset(data_dir:os.PathLike, dataset_name:str, train:bool=True) -> Dataset:
    if dataset_name == 'MNIST':
        dataset = MNIST(data_dir, train=train, download=True, transform=ToTensor())
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_dir, train=train, download=True, transform=ToTensor())
    else:
        # TODO: CIFAR10? Transformer on MS-COCO?
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")
            
    return dataset


def build_dataloader(dataset:Dataset, batch_size:int, sequential:bool=True) -> DataLoader:
    if sequential:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset, replacement=True)
        
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    return dataloader


def spoil_dataset(dataset:Dataset, num_good_points:int, num_bad_points:int, random_seed:int=42) -> Dataset:
    """
    Takes dataset and replaces part of it with shuffled labels
    """
    
    assert num_good_points + num_bad_points == len(dataset)
    
    if num_bad_points == 0:
        return dataset
    
    good_part = Subset(dataset, range(num_good_points))
    bad_part = Subset(dataset, range(num_good_points, len(dataset)))

    # Using the same shuffling for all the runs
    # TODO: should we use the same sampling for different values of num_bad_points?
    shuffling = np.random.RandomState(seed=random_seed).permutation(num_bad_points)
    bad_part_labels = np.array([y for _, y in bad_part])[shuffling]
    bad_part = [(x, bad_y) for (x, _), bad_y in zip(bad_part, bad_part_labels)]

    # Combining good and bad points
    return ConcatDataset([good_part, bad_part])

