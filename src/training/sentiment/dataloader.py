"""
dataloader.py

Creates PyTorch DataLoaders.
"""

import torch

from torch.utils.data import DataLoader

class RetailDataLoaderFactory:

    @staticmethod
    def create(

        dataset,

        batch_size=32,

        shuffle=False,

        num_workers=4,
        pin_memory= False

    ):

        return DataLoader(

            dataset=dataset,

            batch_size=batch_size,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=pin_memory,

            persistent_workers=num_workers > 0,
        )