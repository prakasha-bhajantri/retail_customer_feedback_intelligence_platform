"""
summarizer.py

Executive Summary Generator.

Produces business summaries from analytics.

Supports

- Rule-based summary (offline)
- LLM summary (future)
"""

from __future__ import annotations

from typing import Dict


class ReviewSummarizer:
    """
    Generates executive summaries.
    """

    ###############################################################
    # Rule Based Summary
    ###############################################################

    @staticmethod
    def summarize(
        dashboard: Dict,
    ) -> str:

        summary = dashboard["summary"]

        total = summary.total_reviews

        positive = summary.positive_reviews

        neutral = summary.neutral_reviews

        negative = summary.negative_reviews

        rating = summary.average_rating

        top_positive = dashboard["top_positive_products"][:5]

        top_negative = dashboard["top_negative_products"][:5]

        categories = dashboard["category_insights"][:5]

        lines = []

        ###############################################################
        # Overview
        ###############################################################

        lines.append("EXECUTIVE SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        lines.append(
            f"Total Reviews Analysed : {total:,}"
        )

        lines.append(
            f"Average Rating : {rating:.2f}/5"
        )

        lines.append("")

        ###############################################################
        # Sentiment
        ###############################################################

        positive_pct = (
            positive / total * 100
            if total
            else 0
        )

        negative_pct = (
            negative / total * 100
            if total
            else 0
        )

        lines.append(
            f"Positive Reviews : {positive} ({positive_pct:.1f}%)"
        )

        lines.append(
            f"Neutral Reviews : {neutral}"
        )

        lines.append(
            f"Negative Reviews : {negative} ({negative_pct:.1f}%)"
        )

        lines.append("")

        ###############################################################
        # Best Products
        ###############################################################

        lines.append("TOP POSITIVE PRODUCTS")

        for product in top_positive:

            lines.append(

                f"• {product.product} "

                f"(+{product.positive} / "

                f"-{product.negative})"

            )

        lines.append("")

        ###############################################################
        # Worst Products
        ###############################################################

        lines.append("TOP NEGATIVE PRODUCTS")

        for product in top_negative:

            lines.append(

                f"• {product.product} "

                f"(+{product.positive} / "

                f"-{product.negative})"

            )

        lines.append("")

        ###############################################################
        # Categories
        ###############################################################

        lines.append(
            "MOST DISCUSSED CATEGORIES"
        )

        for category in categories:

            lines.append(

                f"• {category.category} "

                f"({category.mentions} mentions)"

            )

        lines.append("")

        ###############################################################
        # Recommendation
        ###############################################################

        recommendations = []

        if negative_pct > 30:

            recommendations.append(
                "- Investigate quality issues driving customer dissatisfaction."
            )

        if rating < 3.5:

            recommendations.append(
                "- Average rating is low; prioritize product improvement initiatives."
            )

        if not recommendations:

            recommendations.append(
                "- Customer sentiment is generally positive. Continue monitoring emerging issues."
            )

        lines.append("RECOMMENDATIONS")

        for recommendation in recommendations:

            lines.append(recommendation)

        return "\n".join(lines)

    ###############################################################
    # Placeholder for LLM Summary
    ###############################################################

    @staticmethod
    def llm_prompt(
        dashboard: Dict,
    ) -> str:
        """
        Future GenAI integration.
        """

        return f"""
You are a Retail Analytics Expert.

Generate an executive report.

Dashboard Statistics

{dashboard}

Summarize

1. Overall customer sentiment

2. Major customer complaints

3. Top performing products

4. Underperforming products

5. Recommended business actions

Maximum 250 words.
"""