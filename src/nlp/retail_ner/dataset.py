"""
dataset.py

PyTorch Dataset for Retail Named Entity Recognition.

Handles alignment between word-level BIO labels
and BERT subword tokens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.nlp.retail_ner.labels import (
    LABEL2ID,
    IGNORE_INDEX,
)


class RetailNERDataset(Dataset):
    """
    Dataset for Retail Named Entity Recognition.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
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
    # Sample
    ####################################################

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        tokens = row["tokens"]

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        ner_tags = row["ner_tags"]

        if hasattr(ner_tags, "tolist"):
            ner_tags = ner_tags.tolist()

        ####################################################
        # Validation
        ####################################################

        if len(tokens) != len(ner_tags):

            raise ValueError(

                f"Token/Label mismatch: "

                f"{len(tokens)} tokens "

                f"{len(ner_tags)} labels"

            )

        encoding = self.tokenizer(

            tokens,

            is_split_into_words=True,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_offsets_mapping=False,
        )

        ####################################################
        # Word Alignment
        ####################################################

        word_ids = encoding.word_ids()

        labels = []

        previous_word = None

        for word_idx in word_ids:

            ####################################################
            # CLS / SEP / PAD
            ####################################################

            if word_idx is None:

                labels.append(
                    IGNORE_INDEX
                )

                continue

            ####################################################
            # First subword
            ####################################################

            if word_idx != previous_word:

                labels.append(

                    LABEL2ID[
                        ner_tags[word_idx]
                    ]

                )

            ####################################################
            # Remaining subwords
            ####################################################

            else:

                labels.append(
                    IGNORE_INDEX
                )

            previous_word = word_idx

        ####################################################
        # Return Sample
        ####################################################

        return {

            "input_ids":
                torch.tensor(
                    encoding["input_ids"],
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    encoding["attention_mask"],
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
        }