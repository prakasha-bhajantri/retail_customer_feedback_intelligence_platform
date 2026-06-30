"""
prediction_result.py

Prediction result object used by the
Retail Customer Feedback Intelligence Platform.

This dataclass represents the output of a sentiment
prediction for a single customer review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class PredictionResult:
    """
    Represents a sentiment prediction.

    Attributes
    ----------
    label
        Predicted sentiment label.

    label_id
        Integer class id.

    confidence
        Probability of the predicted class.

    probabilities
        Probability distribution across
        all sentiment classes.
    """

    label: str

    label_id: int

    confidence: float

    probabilities: Dict[str, float]

    def to_dict(self) -> Dict:
        """
        Convert prediction to dictionary.

        Useful for:
        - FastAPI responses
        - JSON serialization
        - Vertex AI responses
        """

        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }

    def __str__(self) -> str:
        """
        Human readable representation.
        """

        return (
            f"Prediction("
            f"label='{self.label}', "
            f"confidence={self.confidence:.4f}"
            f")"
        )