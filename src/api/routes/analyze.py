"""
analyze.py

Single Review Analysis API.
"""

from fastapi import APIRouter, HTTPException
import logging

from src.api.schemas import (
    ReviewRequest,
    ReviewAnalysisResponse,
    EntityResponse,
    AspectResponse,
)

from src.pipeline.review_pipeline import ReviewPipeline

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
)

####################################################
# Load Pipeline
####################################################

pipeline = ReviewPipeline(
    sentiment_model_path="artifacts/sentiment/best_model",   # Update path if required
)

####################################################
# Analyze One Review
####################################################

@router.post(
    "",
    response_model=ReviewAnalysisResponse,
)
def analyze(request: ReviewRequest):

    try:

        result = pipeline.analyze_review(

            review_id=request.review_id or "",

            review_text=request.review_text,

            rating=request.rating,

            product_name=request.product_name,

            department=request.department,

            category=request.category,

            subcategory=request.subcategory,

        )

        return ReviewAnalysisResponse(

            review_id=result.review_id,

            sentiment=result.sentiment,

            sentiment_confidence=result.sentiment_confidence,

            entities=[

                EntityResponse(

                    text=e.text,

                    label=e.label,

                    confidence=e.confidence,

                )

                for e in result.entities

            ],

            aspects=[

                AspectResponse(

                    entity=a.entity,

                    entity_type=a.entity_type,

                    sentiment=a.sentiment,

                    confidence=a.confidence,

                )

                for a in result.aspects

            ],

        )

    except Exception as ex:

        logger.exception(ex)

        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )