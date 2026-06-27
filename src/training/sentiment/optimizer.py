"""
optimizer.py

Optimizer factory for Retail Sentiment Training.
"""

import torch


class OptimizerFactory:

    @staticmethod
    def adamw(
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
    ):

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )