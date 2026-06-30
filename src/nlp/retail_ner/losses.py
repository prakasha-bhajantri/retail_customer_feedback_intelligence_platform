"""
losses.py

Loss functions for Retail Named Entity Recognition.
"""

from __future__ import annotations

import torch.nn as nn

from src.nlp.retail_ner.labels import (
    IGNORE_INDEX,
)


class LossFactory:
    """
    Factory for NER loss functions.
    """

    @staticmethod
    def cross_entropy():

        return nn.CrossEntropyLoss(

            ignore_index=IGNORE_INDEX

        )