"""
predictor.py

Inference engine for Retail Sentiment Classification.

Responsibilities
----------------
* Load trained model
* Load tokenizer
* Tokenize input
* Perform inference
* Return PredictionResult
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from src.inference.sentiment.prediction_result import (
    PredictionResult,
)
from src.training.device import get_device


class SentimentPredictor:
    """
    Retail Sentiment Predictor.
    """

    def __init__(
        self,
        model_directory: str | Path,
        max_length: int = 256,
    ):

        self.device = get_device()

        self.max_length = max_length

        self.model_directory = Path(
            model_directory
        )

        ####################################################
        # Load tokenizer
        ####################################################

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_directory,
            )
        )

        ####################################################
        # Load model
        ####################################################

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_directory,
            )
        )

        self.model.to(self.device)

        self.model.eval()

        ####################################################
        # Label Mapping
        ####################################################

        self.id2label = {

            int(key): value

            for key, value in

            self.model.config.id2label.items()

        }

    ########################################################
    # Internal
    ########################################################

    def _tokenize(
        self,
        text: str,
    ):

        return self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt",
        )

    ########################################################
    # Predict
    ########################################################

    def predict(
        self,
        text: str,
    ) -> PredictionResult:

        encoding = self._tokenize(text)

        input_ids = encoding[
            "input_ids"
        ].to(self.device)

        attention_mask = encoding[
            "attention_mask"
        ].to(self.device)

        with torch.no_grad():

            outputs = self.model(

                input_ids=input_ids,

                attention_mask=attention_mask,
            )

        logits = outputs.logits

        probabilities = F.softmax(

            logits,

            dim=1,
        )

        probabilities = (
            probabilities
            .squeeze(0)
            .cpu()
            .numpy()
        )

        label_id = int(
            probabilities.argmax()
        )

        confidence = float(
            probabilities[label_id]
        )

        probability_dict = {

            self.id2label[index]: float(prob)

            for index, prob in enumerate(
                probabilities
            )

        }

        return PredictionResult(

            label=self.id2label[
                label_id
            ],

            label_id=label_id,

            confidence=confidence,

            probabilities=probability_dict,
        )

    ########################################################
    # Batch Prediction
    ########################################################

    def predict_many(
        self,
        texts: List[str],
    ) -> List[PredictionResult]:

        results = []

        for text in texts:

            results.append(
                self.predict(text)
            )

        return results

    ########################################################
    # Probability Only
    ########################################################

    def predict_proba(
        self,
        text: str,
    ) -> dict:

        return self.predict(
            text
        ).probabilities