"""
Home View
"""

from __future__ import annotations

import streamlit as st


def show_home():

    st.title("🛍️ Retail Customer Feedback Intelligence Platform")

    st.caption(
        "Enterprise NLP Analytics for Customer Reviews"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Models",
            "3",
        )

    with col2:

        st.metric(
            "Analytics",
            "5",
        )

    with col3:

        st.metric(
            "APIs",
            "2",
        )

    st.divider()

    st.header("📌 Project Overview")

    st.markdown(
        """
The Retail Customer Feedback Intelligence Platform is an end-to-end NLP
solution that transforms unstructured customer reviews into actionable
business insights.

It combines multiple AI models to automatically detect sentiment,
extract retail entities, identify product aspects, and generate
executive-level summaries for decision-makers.
"""
    )

    st.divider()

    st.header("🚀 Features")

    features = [
        "Sentiment Analysis using BERT",
        "Retail Named Entity Recognition (NER)",
        "Aspect-Based Sentiment Analysis",
        "Business Insight Engine",
        "Executive Summary Generation",
        "Interactive Analytics Dashboard",
        "Single Review Analysis",
        "Batch CSV Processing",
    ]

    for feature in features:
        st.markdown(f"✅ {feature}")

    st.divider()

    st.header("🏗️ Architecture")

    st.code(
        """
Customer Reviews
        │
        ▼
Sentiment Analysis
        │
        ▼
Retail NER
        │
        ▼
Aspect Engine
        │
        ▼
Business Insight Engine
        │
        ▼
Executive Summary
        │
        ▼
Dashboard
        """,
        language="text",
    )

    st.divider()

    st.header("🛠 Technology Stack")

    tech1, tech2 = st.columns(2)

    with tech1:

        st.markdown(
            """
### Backend

- Python
- FastAPI
- PyTorch
- Hugging Face
- Pandas
- NumPy
"""
        )

    with tech2:

        st.markdown(
            """
### Frontend

- Streamlit
- Plotly
- HTML
- CSS
"""
        )

    st.divider()

    st.success("Application Ready ✅")

    st.caption("Retail Customer Feedback Intelligence Platform • Version 1.0")