"""
inference_pipeline.py

Enterprise inference pipeline for the
Retail Customer Feedback Intelligence Platform.

Provides a single entry point for

- Single prediction
- Batch prediction
- DataFrame prediction

This is the class consumed by

- FastAPI
- Vertex AI
- Batch jobs
- Streamlit
- Jupyter notebooks
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.inference.sentiment.prediction_result import (
    PredictionResult,
)
from src.inference.sentiment.predictor_factory import (
    PredictorFactory,
)


class SentimentInferencePipeline:
    """
    Enterprise Sentiment Inference Pipeline.
    """

    def __init__(
        self,
        model_directory: str | Path,
        max_length: int = 256,
        batch_size: int = 64,
    ):

        self.predictor = PredictorFactory.create(
            model_directory=model_directory,
            max_length=max_length,
        )

        self.batch_predictor = (
            PredictorFactory.create_batch(
                model_directory=model_directory,
                max_length=max_length,
                batch_size=batch_size,
            )
        )

    ####################################################
    # Single Prediction
    ####################################################

    def predict(
        self,
        text: str,
    ) -> PredictionResult:

        return self.predictor.predict(
            text
        )

    ####################################################
    # Multiple Prediction
    ####################################################

    def predict_many(
        self,
        texts: List[str],
    ) -> List[PredictionResult]:

        return self.batch_predictor.predict(
            texts
        )

    ####################################################
    # DataFrame Prediction
    ####################################################

    def predict_dataframe(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "review_text",
    ) -> pd.DataFrame:

        return self.batch_predictor.predict_dataframe(
            dataframe,
            text_column=text_column,
        )

    ####################################################
    # Parquet Prediction
    ####################################################

    def predict_parquet(
        self,
        input_file: str | Path,
        output_file: str | Path,
        text_column: str = "review_text",
    ) -> pd.DataFrame:

        dataframe = pd.read_parquet(
            input_file
        )

        predictions = self.predict_dataframe(
            dataframe,
            text_column=text_column,
        )

        predictions.to_parquet(
            output_file,
            index=False,
        )

        return predictions

    ####################################################
    # CSV Prediction
    ####################################################

    def predict_csv(
        self,
        input_file: str | Path,
        output_file: str | Path,
        text_column: str = "review_text",
    ) -> pd.DataFrame:

        dataframe = pd.read_csv(
            input_file
        )

        predictions = self.predict_dataframe(
            dataframe,
            text_column=text_column,
        )

        predictions.to_csv(
            output_file,
            index=False,
        )

        return predictions