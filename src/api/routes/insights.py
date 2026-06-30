"""
insights.py

Retail Analytics API

Provides dashboard and business insight APIs.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.analytics.insight_engine import InsightEngine
from src.analytics.summarizer import ReviewSummarizer

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)

##########################################################
# Temporary In-Memory Store
##########################################################

analytics_data = []

##########################################################
# Register Reviews
##########################################################

@router.post("/load")
def load_reviews(reviews):

    global analytics_data

    analytics_data = reviews

    return {
        "message": f"{len(reviews)} reviews loaded."
    }

##########################################################
# Dashboard
##########################################################

@router.get("/dashboard")
def dashboard():

    dashboard = InsightEngine.dashboard(
        analytics_data
    )

    return dashboard

##########################################################
# Executive Summary
##########################################################

@router.get("/summary")
def executive_summary():

    dashboard = InsightEngine.dashboard(
        analytics_data
    )

    summary = ReviewSummarizer.summarize(
        dashboard
    )

    return {

        "summary": summary

    }

##########################################################
# Products
##########################################################

@router.get("/products")
def products():

    return InsightEngine.product_insights(
        analytics_data
    )

##########################################################
# Categories
##########################################################

@router.get("/categories")
def categories():

    return InsightEngine.category_insights(
        analytics_data
    )

##########################################################
# Aspects
##########################################################

@router.get("/aspects")
def aspects():

    return InsightEngine.aspect_frequency(
        analytics_data
    )