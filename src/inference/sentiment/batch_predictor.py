"""
batch_predictor.py

High-performance batch inference for the
Retail Customer Feedback Intelligence Platform.

Designed for:
- Batch prediction
- Offline scoring
- Vertex AI Batch Prediction
- Large-scale inference
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.inference.sentiment.prediction_result import (
    PredictionResult,
)
from src.training.device import get_device


class BatchSentimentPredictor:
    """
    High-performance sentiment predictor.
    """

    def __init__(
        self,
        model_directory: str | Path,
        max_length: int = 256,
        batch_size: int = 64,
    ):

        self.device = get_device()

        self.batch_size = batch_size

        self.max_length = max_length

        self.model_directory = Path(model_directory)

        ##################################################
        # Load tokenizer
        ##################################################

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_directory
        )

        ##################################################
        # Load model
        ##################################################

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_directory
            )
        )

        self.model.to(self.device)

        self.model.eval()

        ##################################################
        # Label Mapping
        ##################################################

        self.id2label = {

            int(k): v

            for k, v in

            self.model.config.id2label.items()

        }

    ##################################################
    # Predict Batch
    ##################################################

    def predict(
        self,
        texts: List[str],
    ) -> List[PredictionResult]:

        results = []

        with torch.no_grad():

            for start in tqdm(

                range(
                    0,
                    len(texts),
                    self.batch_size,
                ),

                desc="Predicting",

            ):

                batch = texts[
                    start:start+self.batch_size
                ]

                encoding = self.tokenizer(

                    batch,

                    padding=True,

                    truncation=True,

                    max_length=self.max_length,

                    return_tensors="pt",
                )

                input_ids = encoding[
                    "input_ids"
                ].to(self.device)

                attention_mask = encoding[
                    "attention_mask"
                ].to(self.device)

                outputs = self.model(

                    input_ids=input_ids,

                    attention_mask=attention_mask,
                )

                probs = F.softmax(

                    outputs.logits,

                    dim=1,
                )

                probs = probs.cpu().numpy()

                ##################################################

                for row in probs:

                    label_id = int(row.argmax())

                    confidence = float(
                        row[label_id]
                    )

                    probability_dict = {

                        self.id2label[i]: float(p)

                        for i, p in enumerate(
                            row
                        )

                    }

                    results.append(

                        PredictionResult(

                            label=self.id2label[
                                label_id
                            ],

                            label_id=label_id,

                            confidence=confidence,

                            probabilities=probability_dict,
                        )

                    )

        return results

    ##################################################
    # Predict DataFrame
    ##################################################

    def predict_dataframe(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "review_text",
    ) -> pd.DataFrame:

        predictions = self.predict(

            dataframe[text_column]
            .astype(str)
            .tolist()

        )

        output = dataframe.copy()

        output["predicted_sentiment"] = [

            p.label

            for p in predictions

        ]

        output["confidence"] = [

            p.confidence

            for p in predictions

        ]

        output["negative_probability"] = [

            p.probabilities["negative"]

            for p in predictions

        ]

        output["neutral_probability"] = [

            p.probabilities["neutral"]

            for p in predictions

        ]

        output["positive_probability"] = [

            p.probabilities["positive"]

            for p in predictions

        ]

        return output