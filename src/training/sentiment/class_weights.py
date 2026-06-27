"""
class_weights.py

Computes balanced class weights for
Retail Sentiment Classification.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


class ClassWeightCalculator:
    """
    Computes balanced class weights from the
    training dataset.
    """

    LABEL_MAPPING = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    @classmethod
    def compute(
        cls,
        parquet_path: str,
    ) -> Dict[str, float]:
        """
        Returns class weights as a dictionary.
        """

        df = pd.read_parquet(Path(parquet_path))

        labels = (
            df["sentiment_label"]
            .map(cls.LABEL_MAPPING)
            .astype(int)
            .values
        )

        classes = np.unique(labels)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels,
        )

        return {
            label: float(weight)
            for label, weight in zip(
                cls.LABEL_MAPPING.keys(),
                weights,
            )
        }

    @classmethod
    def compute_tensor(
        cls,
        parquet_path: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns class weights as a tensor.

        Order:
        negative
        neutral
        positive
        """

        weights = cls.compute(parquet_path)

        return torch.tensor(
            [
                weights["negative"],
                weights["neutral"],
                weights["positive"],
            ],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def print_weights(
        cls,
        parquet_path: str,
    ) -> None:

        weights = cls.compute(parquet_path)

        print("\nClass Weights")
        print("-" * 40)

        for label, weight in weights.items():
            print(f"{label:<10}: {weight:.4f}")