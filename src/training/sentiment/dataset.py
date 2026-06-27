"""
dataset.py

PyTorch Dataset for Retail Sentiment Analysis.
"""

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class RetailSentimentDataset(Dataset):
    """
    PyTorch Dataset for Retail Sentiment Analysis.
    """

    LABEL_MAPPING = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    def __init__(
        self,
        parquet_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):

        self.df = pd.read_parquet(Path(parquet_path))

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["review_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = self.LABEL_MAPPING[
            row["sentiment_label"]
        ]

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
        }