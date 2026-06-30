"""
review_pipeline.py

End-to-End Review Analysis Pipeline.

Pipeline

Review
   │
   ▼
Sentiment Predictor
   │
   ▼
Retail Entity Extraction
   │
   ▼
Aspect Engine
   │
   ▼
ReviewAnalysis

This is the ONLY class that should call ML models.
"""

from __future__ import annotations

import logging
from typing import List

from src.analytics.aspect_engine import AspectEngine
from src.analytics.models import (
    Entity,
    ReviewAnalysis,
)

# ---------------------------------------------------------
# Sentiment Predictor
# ---------------------------------------------------------

from src.training.sentiment.predictor import (
    RetailSentimentPredictor,
)

# ---------------------------------------------------------
# Rule-based Retail Entity Matcher
# ---------------------------------------------------------

from src.nlp.retail_ner.matcher import (
    RetailEntityMatcher,
)

logger = logging.getLogger(__name__)


class ReviewPipeline:
    """
    Enterprise Review Analysis Pipeline.

    Loads NLP models once and reuses them.
    """

    def __init__(
        self,
        sentiment_model_path: str,
    ):

        logger.info(
            "Loading Sentiment Model..."
        )

        self.sentiment_predictor = RetailSentimentPredictor(
            sentiment_model_path
        )

        logger.info(
            "Review Pipeline Ready."
        )

    ###########################################################
    # Sentiment
    ###########################################################

    def predict_sentiment(
        self,
        review: str,
    ):

        result = self.sentiment_predictor.predict(
            review
        )

        return result

    ###########################################################
    # Entity Extraction
    ###########################################################

    def extract_entities(
        self,
        review: str,
        product_name: str = "",
        department: str = "",
        category: str = "",
        subcategory: str = "",
    ) -> List[Entity]:

        tokens, labels = RetailEntityMatcher.label_review(

            review_text=review,

            product_name=product_name,

            department=department,

            category=category,

            subcategory=subcategory,
        )

        entities = []

        current_tokens = []

        current_type = None

        for token, label in zip(tokens, labels):

            if label == "O":

                if current_tokens:

                    entities.append(

                        Entity(

                            text=" ".join(
                                current_tokens
                            ),

                            label=current_type,

                            confidence=1.0,

                        )

                    )

                    current_tokens = []

                    current_type = None

                continue

            prefix, entity = label.split(
                "-",
                maxsplit=1,
            )

            if prefix == "B":

                if current_tokens:

                    entities.append(

                        Entity(

                            text=" ".join(
                                current_tokens
                            ),

                            label=current_type,

                            confidence=1.0,

                        )

                    )

                current_tokens = [token]

                current_type = entity

            elif prefix == "I":

                current_tokens.append(
                    token
                )

        if current_tokens:

            entities.append(

                Entity(

                    text=" ".join(
                        current_tokens
                    ),

                    label=current_type,

                    confidence=1.0,

                )

            )

        return entities

    ###########################################################
    # Single Review
    ###########################################################

    def analyze_review(
        self,
        review_id: str,
        review_text: str,
        rating: int | None = None,
        product_name: str = "",
        department: str = "",
        category: str = "",
        subcategory: str = "",
    ) -> ReviewAnalysis:

        ####################################################
        # Sentiment
        ####################################################

        sentiment = self.predict_sentiment(
            review_text
        )

        ####################################################
        # Entities
        ####################################################

        entities = self.extract_entities(

            review=review_text,

            product_name=product_name,

            department=department,

            category=category,

            subcategory=subcategory,
        )

        ####################################################
        # Aspects
        ####################################################

        aspects = AspectEngine.extract(

            review=review_text,

            sentiment=sentiment["label"],

            sentiment_confidence=sentiment[
                "confidence"
            ],

            entities=entities,
        )

        ####################################################
        # Final Object
        ####################################################

        return ReviewAnalysis(

            review_id=review_id,

            review_text=review_text,

            rating=rating,

            sentiment=sentiment["label"],

            sentiment_confidence=sentiment[
                "confidence"
            ],

            entities=entities,

            aspects=aspects,
        )

    ###########################################################
    # Batch Reviews
    ###########################################################

    def analyze_batch(
        self,
        reviews,
    ) -> List[ReviewAnalysis]:

        results = []

        for review in reviews:

            results.append(

                self.analyze_review(

                    review_id=str(
                        review.get(
                            "review_id",
                            "",
                        )
                    ),

                    review_text=review[
                        "review_text"
                    ],

                    rating=review.get(
                        "rating"
                    ),

                    product_name=review.get(
                        "product_name",
                        "",
                    ),

                    department=review.get(
                        "department",
                        "",
                    ),

                    category=review.get(
                        "category",
                        "",
                    ),

                    subcategory=review.get(
                        "subcategory",
                        "",
                    ),
                )

            )

        return results