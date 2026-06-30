"""
insight_engine.py

Business Insight Engine for the Retail Customer Feedback
Intelligence Platform.

Consumes ReviewAnalysis objects and generates
business-ready insights for dashboards and APIs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List

from src.analytics.models import (
    Aspect,
    DashboardSummary,
    ProductInsight,
    CategoryInsight,
    ReviewAnalysis,
)


class InsightEngine:
    """
    Business Insight Engine.
    """

    ###########################################################
    # Dashboard Summary
    ###########################################################

    @staticmethod
    def dashboard_summary(
        reviews: List[ReviewAnalysis],
    ) -> DashboardSummary:

        sentiments = Counter(
            review.sentiment.lower()
            for review in reviews
        )

        ratings = [

            review.rating

            for review in reviews

            if review.rating is not None

        ]

        return DashboardSummary(

            total_reviews=len(reviews),

            positive_reviews=sentiments.get(
                "positive",
                0,
            ),

            neutral_reviews=sentiments.get(
                "neutral",
                0,
            ),

            negative_reviews=sentiments.get(
                "negative",
                0,
            ),

            average_rating=round(
                mean(ratings),
                2,
            ) if ratings else 0.0,
        )

    ###########################################################
    # Sentiment Distribution
    ###########################################################

    @staticmethod
    def sentiment_distribution(
        reviews: List[ReviewAnalysis],
    ) -> Dict:

        counter = Counter(

            review.sentiment

            for review in reviews

        )

        return dict(counter)

    ###########################################################
    # Rating Distribution
    ###########################################################

    @staticmethod
    def rating_distribution(
        reviews: List[ReviewAnalysis],
    ) -> Dict:

        counter = Counter(

            review.rating

            for review in reviews

            if review.rating is not None

        )

        return dict(counter)

    ###########################################################
    # Product Insights
    ###########################################################

    @staticmethod
    def product_insights(
        reviews: List[ReviewAnalysis],
    ) -> List[ProductInsight]:

        products = defaultdict(

            lambda: {

                "positive": 0,

                "neutral": 0,

                "negative": 0,

                "mentions": 0,

            }

        )

        score_map = {

            "positive": 1,

            "neutral": 0,

            "negative": -1,

        }

        scores = defaultdict(list)

        for review in reviews:

            for aspect in review.aspects:

                if aspect.entity_type != "PRODUCT":

                    continue

                products[aspect.entity]["mentions"] += 1

                products[aspect.entity][

                    aspect.sentiment

                ] += 1

                scores[aspect.entity].append(

                    score_map.get(

                        aspect.sentiment,

                        0,

                    )

                )

        results = []

        for product, stats in products.items():

            results.append(

                ProductInsight(

                    product=product,

                    total_mentions=stats["mentions"],

                    positive=stats["positive"],

                    neutral=stats["neutral"],

                    negative=stats["negative"],

                    average_sentiment=round(

                        mean(scores[product]),

                        2,

                    ),

                )

            )

        return sorted(

            results,

            key=lambda x: x.total_mentions,

            reverse=True,

        )

    ###########################################################
    # Category Insights
    ###########################################################

    @staticmethod
    def category_insights(
        reviews: List[ReviewAnalysis],
    ) -> List[CategoryInsight]:

        categories = defaultdict(

            lambda: {

                "positive": 0,

                "neutral": 0,

                "negative": 0,

                "mentions": 0,

            }

        )

        for review in reviews:

            for aspect in review.aspects:

                if aspect.entity_type not in (

                    "CATEGORY",

                    "SUBCATEGORY",

                ):

                    continue

                categories[aspect.entity][

                    "mentions"

                ] += 1

                categories[aspect.entity][

                    aspect.sentiment

                ] += 1

        results = []

        for category, stats in categories.items():

            results.append(

                CategoryInsight(

                    category=category,

                    mentions=stats["mentions"],

                    positive=stats["positive"],

                    neutral=stats["neutral"],

                    negative=stats["negative"],

                )

            )

        return sorted(

            results,

            key=lambda x: x.mentions,

            reverse=True,

        )

    ###########################################################
    # Top Positive Products
    ###########################################################

    @classmethod
    def top_positive_products(
        cls,
        reviews,
        top_n=10,
    ):

        insights = cls.product_insights(
            reviews
        )

        insights.sort(

            key=lambda x: (

                x.positive,

                x.average_sentiment,

            ),

            reverse=True,

        )

        return insights[:top_n]

    ###########################################################
    # Top Negative Products
    ###########################################################

    @classmethod
    def top_negative_products(
        cls,
        reviews,
        top_n=10,
    ):

        insights = cls.product_insights(
            reviews
        )

        insights.sort(

            key=lambda x: (

                x.negative,

                -x.average_sentiment,

            ),

            reverse=True,

        )

        return insights[:top_n]

    ###########################################################
    # Most Mentioned Aspects
    ###########################################################

    @staticmethod
    def aspect_frequency(
        reviews: List[ReviewAnalysis],
    ) -> Dict:

        counter = Counter()

        for review in reviews:

            for aspect in review.aspects:

                counter[aspect.entity] += 1

        return dict(

            counter.most_common(20)

        )

    ###########################################################
    # Complete Dashboard
    ###########################################################

    @classmethod
    def dashboard(
        cls,
        reviews: List[ReviewAnalysis],
    ) -> Dict:

        return {

            "summary":

                cls.dashboard_summary(
                    reviews
                ),

            "sentiment_distribution":

                cls.sentiment_distribution(
                    reviews
                ),

            "rating_distribution":

                cls.rating_distribution(
                    reviews
                ),

            "top_positive_products":

                cls.top_positive_products(
                    reviews
                ),

            "top_negative_products":

                cls.top_negative_products(
                    reviews
                ),

            "category_insights":

                cls.category_insights(
                    reviews
                ),

            "aspect_frequency":

                cls.aspect_frequency(
                    reviews
                ),
        }