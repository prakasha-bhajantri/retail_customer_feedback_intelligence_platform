"""
predictor_factory.py

Factory for creating sentiment inference engines.

Provides centralized construction of:

- SentimentPredictor
- BatchSentimentPredictor

This allows the rest of the application to remain
independent of the underlying model implementation.
"""

from __future__ import annotations

from pathlib import Path

from src.inference.sentiment.batch_predictor import (
    BatchSentimentPredictor,
)
from src.inference.sentiment.predictor import (
    SentimentPredictor,
)


class PredictorFactory:
    """
    Factory class responsible for creating
    sentiment predictors.
    """

    @staticmethod
    def create(
        model_directory: str | Path,
        max_length: int = 256,
    ) -> SentimentPredictor:
        """
        Create predictor for single-review inference.
        """

        return SentimentPredictor(
            model_directory=model_directory,
            max_length=max_length,
        )

    @staticmethod
    def create_batch(
        model_directory: str | Path,
        max_length: int = 256,
        batch_size: int = 64,
    ) -> BatchSentimentPredictor:
        """
        Create predictor for batch inference.
        """

        return BatchSentimentPredictor(
            model_directory=model_directory,
            max_length=max_length,
            batch_size=batch_size,
        )