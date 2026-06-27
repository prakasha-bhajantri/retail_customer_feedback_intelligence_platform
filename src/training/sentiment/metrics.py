"""
metrics.py

Evaluation metrics for sentiment classification.
"""

from typing import Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


class SentimentMetrics:

    @staticmethod
    def compute(
        labels,
        predictions,
    ) -> Dict:

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        return {

            "accuracy": round(float(accuracy), 4),

            "precision": round(float(precision), 4),

            "recall": round(float(recall), 4),

            "f1": round(float(f1), 4),
        }

    @staticmethod
    def classification_report(
        labels,
        predictions,
    ):

        return classification_report(
            labels,
            predictions,
            target_names=[
                "negative",
                "neutral",
                "positive",
            ],
            digits=4,
            zero_division=0,
        )


    @staticmethod
    def confusion_matrix(
        labels,
        predictions,
    ):

        return confusion_matrix(
            labels,
            predictions,
            labels=[0, 1, 2],
        )