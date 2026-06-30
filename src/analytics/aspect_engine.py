"""
aspect_engine.py

Aspect Sentiment Engine.

Combines Sentiment + NER outputs to produce
aspect-level sentiment.

V1 Strategy
-----------
Sentence Level Sentiment Attribution

Future
------
Replace with Aspect-Based Sentiment Transformer.
"""

from __future__ import annotations

import re

from typing import List

from src.analytics.models import (
    Aspect,
    Entity,
)


class AspectEngine:
    """
    Generates aspect sentiment.

    Input

    Review
        ↓

    Sentiment

    +

    NER Entities

        ↓

    Aspect Sentiment
    """

    ####################################################
    # Sentence Split
    ####################################################

    @staticmethod
    def split_sentences(
        review: str,
    ) -> List[str]:

        if not review:

            return []

        sentences = re.split(

            r"[.!?]+",

            review,

        )

        return [

            sentence.strip()

            for sentence in sentences

            if sentence.strip()

        ]

    ####################################################
    # Entity belongs to sentence?
    ####################################################

    @staticmethod
    def entity_in_sentence(

        entity: Entity,

        sentence: str,

    ) -> bool:

        return (

            entity.text.lower()

            in

            sentence.lower()

        )

    ####################################################
    # Public API
    ####################################################

    @classmethod
    def extract(

        cls,

        review: str,

        sentiment: str,

        sentiment_confidence: float,

        entities: List[Entity],

    ) -> List[Aspect]:

        aspects = []

        if not entities:

            return aspects

        sentences = cls.split_sentences(
            review
        )

        ####################################################
        # No sentence split
        ####################################################

        if len(sentences) == 0:

            for entity in entities:

                aspects.append(

                    Aspect(

                        entity=entity.text,

                        entity_type=entity.label,

                        sentiment=sentiment,

                        confidence=min(

                            entity.confidence,

                            sentiment_confidence,

                        ),

                    )

                )

            return aspects

        ####################################################
        # Sentence Attribution
        ####################################################

        for entity in entities:

            assigned = False

            for sentence in sentences:

                if cls.entity_in_sentence(

                    entity,

                    sentence,

                ):

                    aspects.append(

                        Aspect(

                            entity=entity.text,

                            entity_type=entity.label,

                            sentiment=sentiment,

                            confidence=min(

                                entity.confidence,

                                sentiment_confidence,

                            ),

                        )

                    )

                    assigned = True

                    break

            ####################################################
            # Fallback
            ####################################################

            if not assigned:

                aspects.append(

                    Aspect(

                        entity=entity.text,

                        entity_type=entity.label,

                        sentiment=sentiment,

                        confidence=min(

                            entity.confidence,

                            sentiment_confidence,

                        ),

                    )

                )

        return aspects