"""
cards.py

Reusable KPI cards.
"""

import streamlit as st


def metric_cards(summary):

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "📋 Reviews",
        summary.total_reviews,
    )

    col2.metric(
        "⭐ Avg Rating",
        round(summary.average_rating, 2),
    )

    col3.metric(
        "😊 Positive",
        summary.positive_reviews,
    )

    col4.metric(
        "😐 Neutral",
        summary.neutral_reviews,
    )

    col5.metric(
        "😞 Negative",
        summary.negative_reviews,
    )