"""
predictor.py

Inference module for Retail Sentiment Analysis.
"""

from __future__ import annotations

import torch

from transformers import AutoTokenizer

from src.training.device import get_device
from src.training.sentiment.config import SentimentConfig
from src.training.sentiment.model_factory import ModelFactory


class RetailSentimentPredictor:
    """
    Retail Sentiment Predictor.
    """

    def __init__(
        self,
        model_path=None,
    ):

        if model_path is None:

            model_path = SentimentConfig.BEST_MODEL

        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            SentimentConfig.MODEL_NAME
        )

        self.model = ModelFactory.load(
            model_path
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    ############################################################
    # Predict
    ############################################################

    @torch.no_grad()
    def predict(
        self,
        text: str,
    ):

        encoding = self.tokenizer(

            text,

            truncation=True,

            padding=True,

            max_length=SentimentConfig.MAX_LENGTH,

            return_tensors="pt",

        )

        encoding = {

            key: value.to(self.device)

            for key, value in encoding.items()

        }

        outputs = self.model(

            **encoding

        )

        probabilities = torch.softmax(

            outputs.logits,

            dim=1,

        )

        confidence, prediction = torch.max(

            probabilities,

            dim=1,

        )

        prediction = prediction.item()

        confidence = confidence.item()

        return {

            "label":

                SentimentConfig.ID2LABEL[
                    prediction
                ],

            "confidence":

                round(
                    confidence,
                    4,
                ),

            "probabilities":

                probabilities.squeeze()
                .cpu()
                .tolist(),

        }