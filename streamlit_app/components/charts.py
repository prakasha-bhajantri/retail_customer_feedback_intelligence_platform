"""
charts.py

Reusable Plotly charts.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st


####################################################
# Sentiment Distribution
####################################################

def sentiment_chart(distribution: dict):

    if not distribution:
        st.info("No sentiment data available.")
        return

    fig = px.pie(
        names=list(distribution.keys()),
        values=list(distribution.values()),
        title="Sentiment Distribution",
        hole=0.45,
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(
        fig,
        # use_container_width=True,
        width="stretch",
    )


####################################################
# Rating Distribution
####################################################

def rating_chart(distribution: dict):

    if not distribution:
        st.info("No rating data available.")
        return

    fig = px.bar(
        x=list(distribution.keys()),
        y=list(distribution.values()),
        labels={
            "x": "Rating",
            "y": "Reviews",
        },
        title="Rating Distribution",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(
        fig,
        # use_container_width=True,
        width="stretch",
    )


####################################################
# Aspect Frequency
####################################################

def aspect_chart(aspects: dict):

    if not aspects:
        st.info("No aspects detected.")
        return

    fig = px.bar(
        x=list(aspects.keys()),
        y=list(aspects.values()),
        labels={
            "x": "Aspect",
            "y": "Mentions",
        },
        title="Aspect Frequency",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(
        fig,
        # use_container_width=True,
        width="stretch",
    )