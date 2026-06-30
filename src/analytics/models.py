"""
models.py

Shared dataclasses for the Analytics module.

These classes are used by

- Aspect Engine
- Insight Engine
- Summarizer
- FastAPI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


###########################################################
# NER Entity
###########################################################

@dataclass(slots=True)
class Entity:
    """
    Named Entity extracted from a review.
    """

    text: str

    label: str

    confidence: float


###########################################################
# Aspect
###########################################################

@dataclass(slots=True)
class Aspect:
    """
    Aspect-level sentiment.
    """

    entity: str

    entity_type: str

    sentiment: str

    confidence: float = 1.0


###########################################################
# Review Analysis
###########################################################

@dataclass(slots=True)
class ReviewAnalysis:
    """
    Complete NLP output for one review.
    """

    review_id: str

    review_text: str

    rating: Optional[int]

    sentiment: str

    sentiment_confidence: float

    entities: List[Entity] = field(default_factory=list)

    aspects: List[Aspect] = field(default_factory=list)


###########################################################
# Dashboard Summary
###########################################################

@dataclass(slots=True)
class DashboardSummary:

    total_reviews: int

    positive_reviews: int

    neutral_reviews: int

    negative_reviews: int

    average_rating: float


###########################################################
# Product Insight
###########################################################

@dataclass(slots=True)
class ProductInsight:

    product: str

    total_mentions: int

    positive: int

    neutral: int

    negative: int

    average_sentiment: float


###########################################################
# Category Insight
###########################################################

@dataclass(slots=True)
class CategoryInsight:

    category: str

    mentions: int

    positive: int

    neutral: int

    negative: int


###########################################################
# Executive Summary
###########################################################

@dataclass(slots=True)
class ExecutiveSummary:

    summary: str

    top_positive_products: List[str]

    top_negative_products: List[str]

    top_categories: List[str]

    recommendations: List[str]