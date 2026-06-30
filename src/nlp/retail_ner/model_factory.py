"""
model_factory.py

Factory for creating Hugging Face Token Classification
models used by Retail Named Entity Recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch.nn as nn

from transformers import (

    AutoConfig,

    AutoModelForTokenClassification,

)


class ModelFactory:
    """
    Factory for Retail NER models.

    Supports

    - BERT
    - RoBERTa
    - DeBERTa
    - DistilBERT
    - ModernBERT
    """

    @staticmethod
    def create(

        model_name: str,

        num_labels: int,

        id2label: Optional[dict] = None,

        label2id: Optional[dict] = None,

    ) -> nn.Module:

        config = AutoConfig.from_pretrained(

            model_name,

            num_labels=num_labels,

            id2label=id2label,

            label2id=label2id,

        )

        model = AutoModelForTokenClassification.from_pretrained(

            model_name,

            config=config,

        )

        return model

    @staticmethod
    def load(
        model_directory: str | Path,
    ):

        return AutoModelForTokenClassification.from_pretrained(

            model_directory

        )

    @staticmethod
    def save(

        model: nn.Module,

        output_directory: str | Path,

    ) -> None:

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        model.save_pretrained(
            output_directory
        )