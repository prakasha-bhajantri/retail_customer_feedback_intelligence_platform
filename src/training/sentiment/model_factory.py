"""
model_factory.py

Factory for creating and loading Hugging Face sequence
classification models used by the Retail Customer
Feedback Intelligence Platform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


class ModelFactory:
    """
    Factory class responsible for creating transformer
    sequence classification models.

    Supports any Hugging Face transformer model
    compatible with AutoModelForSequenceClassification.

    Examples
    --------
    >>> model = ModelFactory.create(
    ...     "bert-base-uncased",
    ...     num_labels=3
    ... )
    """

    @staticmethod
    def create(
        model_name: str,
        num_labels: int,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
    ) -> nn.Module:
        """
        Create a new transformer model.

        Parameters
        ----------
        model_name
            Hugging Face model name.

        num_labels
            Number of output classes.

        id2label
            Optional label mapping.

        label2id
            Optional reverse label mapping.

        Returns
        -------
        nn.Module
        """

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

        return model

    @staticmethod
    def load(model_directory: str | Path):
        """
        Load a previously trained model.

        Parameters
        ----------
        model_directory
            Directory containing save_pretrained() output.

        Returns
        -------
        nn.Module
        """

        return AutoModelForSequenceClassification.from_pretrained(
            model_directory
        )

    @staticmethod
    def save(
        model: nn.Module,
        output_directory: str | Path,
    ) -> None:
        """
        Save a Hugging Face model.

        Parameters
        ----------
        model
            Trained transformer model.

        output_directory
            Destination directory.
        """

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(output_directory)