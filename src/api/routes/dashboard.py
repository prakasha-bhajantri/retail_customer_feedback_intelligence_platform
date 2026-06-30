"""
dashboard.py

Batch Review Dashboard API.

Runs the complete analytics pipeline and returns
dashboard-ready insights.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi import HTTPException

from dataclasses import asdict

from src.api.schemas import (
    BatchReviewRequest,
    DashboardResponse,
)

from src.analytics.insight_engine import (
    InsightEngine,
)

from src.analytics.summarizer import (
    ReviewSummarizer,
)

from src.pipeline.review_pipeline import (
    ReviewPipeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)

###############################################################
# Load Pipeline Once
###############################################################

pipeline = ReviewPipeline(
    sentiment_model_path="artifacts/sentiment/best_model",
)

###############################################################
# Dashboard
###############################################################

@router.post(
    "",
    response_model=DashboardResponse,
)
def dashboard(
    request: BatchReviewRequest,
):

    try:

        ####################################################
        # Convert Pydantic -> dict
        ####################################################

        reviews = [

            review.model_dump()

            for review in request.reviews

        ]

        ####################################################
        # NLP Pipeline
        ####################################################

        analyses = pipeline.analyze_batch(
            reviews
        )

        ####################################################
        # Analytics
        ####################################################

        dashboard = InsightEngine.dashboard(
            analyses
        )

        ####################################################
        # Executive Summary
        ####################################################

        summary = ReviewSummarizer.summarize(
            dashboard
        )

        ####################################################
        # Convert dataclasses
        ####################################################

        dashboard["summary"] = asdict(
            dashboard["summary"]
        )

        dashboard["top_positive_products"] = [
            asdict(product)
            for product in dashboard["top_positive_products"]
        ]

        dashboard["top_negative_products"] = [
            asdict(product)
            for product in dashboard["top_negative_products"]
        ]

        dashboard["category_insights"] = [
            asdict(category)
            for category in dashboard["category_insights"]
        ]

        dashboard["executive_summary"] = summary

        return DashboardResponse(
            **dashboard
        )

    except Exception as ex:

        logger.exception(ex)

        raise HTTPException(

            status_code=500,

            detail=str(ex),

        )