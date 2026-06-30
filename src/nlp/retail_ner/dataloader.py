"""
dataloader.py

Creates DataLoaders for Retail NER.
"""

from __future__ import annotations

import torch

from torch.utils.data import DataLoader


class RetailNERDataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders
    used during Retail NER training.
    """

    @staticmethod
    def create(
        dataset,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):

        return DataLoader(

            dataset=dataset,

            batch_size=batch_size,

            shuffle=shuffle,

            num_workers=num_workers,

            pin_memory=(
                pin_memory
                and torch.cuda.is_available()
            ),

            persistent_workers=(
                num_workers > 0
            ),
        )