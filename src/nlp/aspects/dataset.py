"""
dataset.py

PyTorch Dataset for Retail Aspect Detection.

Each review may contain multiple retail aspects.

The dataset converts aspect labels into a
multi-hot encoded vector suitable for
multi-label classification.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.nlp.aspects.labels import (
    LABEL2ID,
    NUM_LABELS,
)


class RetailAspectDataset(Dataset):
    """
    Dataset for Retail Aspect Detection.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):

        self.df = pd.read_parquet(
            Path(parquet_path)
        )

        self.tokenizer = tokenizer

        self.max_length = max_length

    ####################################################
    # Dataset Length
    ####################################################

    def __len__(self):

        return len(self.df)

    ####################################################
    # Multi-hot Encoder
    ####################################################

    @staticmethod
    def encode_labels(
        aspects: str,
    ) -> torch.Tensor:
        """
        Converts

        battery,packaging

        into

        tensor([
            1,
            0,
            ...
            1
        ])
        """

        labels = torch.zeros(
            NUM_LABELS,
            dtype=torch.float32,
        )

        if (
            aspects is None
            or aspects == ""
        ):
            return labels

        for aspect in aspects.split(","):

            aspect = aspect.strip()

            if aspect in LABEL2ID:

                labels[
                    LABEL2ID[aspect]
                ] = 1.0

        return labels

    ####################################################
    # Sample
    ####################################################

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        encoding = self.tokenizer(

            row["review_text"],

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt",
        )

        labels = self.encode_labels(
            row["aspects"]
        )

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                labels,
        }