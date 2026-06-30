"""
schemas.py

Pydantic request/response models for
Retail Customer Feedback Intelligence Platform.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


###############################################################
# Input Models
###############################################################

class ReviewRequest(BaseModel):
    """
    Single review request.
    """

    review_id: Optional[str] = None

    review_text: str = Field(
        ...,
        min_length=1,
    )

    rating: Optional[int] = None

    product_name: Optional[str] = ""

    department: Optional[str] = ""

    category: Optional[str] = ""

    subcategory: Optional[str] = ""


class BatchReviewRequest(BaseModel):
    """
    Batch Review Request
    """

    reviews: List[ReviewRequest]


###############################################################
# Entity
###############################################################

class EntityResponse(BaseModel):

    text: str

    label: str

    confidence: float


###############################################################
# Aspect
###############################################################

class AspectResponse(BaseModel):

    entity: str

    entity_type: str

    sentiment: str

    confidence: float


###############################################################
# Single Review Response
###############################################################

class ReviewAnalysisResponse(BaseModel):

    review_id: str

    sentiment: str

    sentiment_confidence: float

    entities: List[EntityResponse]

    aspects: List[AspectResponse]


###############################################################
# Dashboard Response
###############################################################

class DashboardResponse(BaseModel):

    summary: dict

    sentiment_distribution: dict

    rating_distribution: dict

    top_positive_products: list

    top_negative_products: list

    category_insights: list

    aspect_frequency: dict

    executive_summary: str