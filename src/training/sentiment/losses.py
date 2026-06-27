"""
losses.py

Loss functions used by the
Retail Sentiment Training Pipeline.
"""

from typing import Optional

import torch
import torch.nn as nn


class LossFactory:
    """
    Factory class for creating loss functions.
    """

    @staticmethod
    def cross_entropy(
        class_weights: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Standard Cross Entropy Loss.

        Parameters
        ----------
        class_weights:
            Tensor containing class weights.
            Shape: [num_classes]
        """

        return nn.CrossEntropyLoss(
            weight=class_weights
        )

    @staticmethod
    def label_smoothing(
        class_weights: Optional[torch.Tensor] = None,
        smoothing: float = 0.1,
    ) -> nn.Module:
        """
        Cross Entropy with Label Smoothing.
        """

        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=smoothing,
        )