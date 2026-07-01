"""
Dashboard View
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.pipeline.review_pipeline import ReviewPipeline
from src.analytics.insight_engine import InsightEngine
from src.analytics.summarizer import ReviewSummarizer

from streamlit_app.config import SENTIMENT_MODEL
from streamlit_app.components.cards import metric_cards
from streamlit_app.components.charts import (
    sentiment_chart,
    rating_chart,
    aspect_chart,
)


@st.cache_resource
def load_pipeline():

    return ReviewPipeline(
        sentiment_model_path=SENTIMENT_MODEL
    )


pipeline = load_pipeline()


def show_dashboard():

    st.title("📊 Analytics Dashboard")

    st.caption(
        "Batch Customer Review Analytics"
    )

    st.divider()

    uploaded = st.file_uploader(

        "Upload Review CSV",

        type=["csv"],

    )

    if uploaded is None:

        st.info(
            "Upload a CSV file to begin analysis."
        )

        return

    ####################################################
    # Read CSV
    ####################################################

    reviews = pd.read_csv(uploaded)

    st.success(
        f"{len(reviews):,} reviews loaded."
    )

    ####################################################
    # Analyze
    ####################################################

    with st.spinner("Running NLP Pipeline..."):

        analyses = pipeline.analyze_batch(

            reviews.to_dict(
                orient="records"
            )

        )

        dashboard = InsightEngine.dashboard(
            analyses
        )

        summary = ReviewSummarizer.summarize(
            dashboard
        )

    ####################################################
    # KPI
    ####################################################

    metric_cards(
        dashboard["summary"]
    )

    st.divider()

    ####################################################
    # Charts
    ####################################################

    col1, col2 = st.columns(2)

    with col1:

        sentiment_chart(

            dashboard[
                "sentiment_distribution"
            ]

        )

    with col2:

        rating_chart(

            dashboard[
                "rating_distribution"
            ]

        )

    st.divider()

    aspect_chart(

        dashboard[
            "aspect_frequency"
        ]

    )

    ####################################################
    # Products
    ####################################################

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🏆 Top Positive Products"
        )

        if dashboard[
            "top_positive_products"
        ]:

            st.dataframe(

                pd.DataFrame(

                    [

                        asdict(x)

                        for x in dashboard[
                            "top_positive_products"
                        ]

                    ]

                ),

                # use_container_width=True,
                width="stretch",

                hide_index=True,

            )

        else:

            st.info(
                "No positive products."
            )

    with right:

        st.subheader(
            "⚠️ Top Negative Products"
        )

        if dashboard[
            "top_negative_products"
        ]:

            st.dataframe(

                pd.DataFrame(

                    [

                        asdict(x)

                        for x in dashboard[
                            "top_negative_products"
                        ]

                    ]

                ),

                # use_container_width=True,
                width="stretch",

                hide_index=True,

            )

        else:

            st.info(
                "No negative products."
            )

    ####################################################
    # Categories
    ####################################################

    if dashboard[
        "category_insights"
    ]:

        st.divider()

        st.subheader(
            "📦 Category Insights"
        )

        st.dataframe(

            pd.DataFrame(

                [

                    asdict(x)

                    for x in dashboard[
                        "category_insights"
                    ]

                ]

            ),

            # use_container_width=True,
            width="stretch",

            hide_index=True,

        )

    ####################################################
    # Summary
    ####################################################

    st.divider()

    st.subheader(
        "📝 Executive Summary"
    )

    st.code(
        summary,
        language="text",
    )

    ####################################################
    # Download
    ####################################################

    st.download_button(

        label="⬇ Download Uploaded CSV",

        data=reviews.to_csv(index=False),

        file_name="reviews.csv",

        mime="text/csv",

    )