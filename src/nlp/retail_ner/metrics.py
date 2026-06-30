"""
metrics.py

Evaluation metrics for Retail Named Entity Recognition.
"""

from __future__ import annotations

from typing import Dict, List

from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.nlp.retail_ner.labels import (
    ID2LABEL,
    IGNORE_INDEX,
)


class NERMetrics:
    """
    Computes entity-level evaluation metrics
    using seqeval.
    """

    @staticmethod
    def decode(
        labels,
        predictions,
    ):
        """
        Converts integer IDs into BIO labels.

        Removes ignored tokens (-100).
        """

        true_labels = []
        predicted_labels = []

        for label_sequence, prediction_sequence in zip(
            labels,
            predictions,
        ):

            true_sentence = []
            predicted_sentence = []

            for label_id, prediction_id in zip(
                label_sequence,
                prediction_sequence,
            ):

                if label_id == IGNORE_INDEX:
                    continue

                true_sentence.append(
                    ID2LABEL[int(label_id)]
                )

                predicted_sentence.append(
                    ID2LABEL[int(prediction_id)]
                )

            true_labels.append(
                true_sentence
            )

            predicted_labels.append(
                predicted_sentence
            )

        return (
            true_labels,
            predicted_labels,
        )

    ####################################################
    # Compute Metrics
    ####################################################

    @classmethod
    def compute(
        cls,
        labels,
        predictions,
    ) -> Dict:

        labels, predictions = cls.decode(
            labels,
            predictions,
        )

        return {

            "accuracy": round(
                accuracy_score(
                    labels,
                    predictions,
                ),
                4,
            ),

            "precision": round(
                precision_score(
                    labels,
                    predictions,
                ),
                4,
            ),

            "recall": round(
                recall_score(
                    labels,
                    predictions,
                ),
                4,
            ),

            "f1": round(
                f1_score(
                    labels,
                    predictions,
                ),
                4,
            ),
        }

    ####################################################
    # Classification Report
    ####################################################

    @classmethod
    def classification_report(
        cls,
        labels,
        predictions,
    ):

        labels, predictions = cls.decode(
            labels,
            predictions,
        )

        return classification_report(
            labels,
            predictions,
            digits=4,
        )