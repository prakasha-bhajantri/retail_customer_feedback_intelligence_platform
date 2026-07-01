"""
Reusable tables for Streamlit.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st


###########################################################
# Generic Table
###########################################################

def dataframe(data):

    if not data:

        st.info("No data available.")

        return

    if isinstance(data, list):

        rows = []

        for item in data:

            try:

                rows.append(asdict(item))

            except Exception:

                rows.append(item.__dict__)

        df = pd.DataFrame(rows)

    else:

        df = pd.DataFrame(data)

    st.dataframe(

        df,

        # use_container_width=True,
        width="stretch",

        hide_index=True,

    )


###########################################################
# Positive Products
###########################################################

def positive_products(products):

    st.subheader("🏆 Top Positive Products")

    dataframe(products)


###########################################################
# Negative Products
###########################################################

def negative_products(products):

    st.subheader("⚠️ Top Negative Products")

    dataframe(products)


###########################################################
# Category Insights
###########################################################

def category_insights(categories):

    st.subheader("📦 Category Insights")

    dataframe(categories)


###########################################################
# Aspect Table
###########################################################

def aspects(aspects):

    st.subheader("🎯 Aspect Sentiment")

    dataframe(aspects)


###########################################################
# Entity Table
###########################################################

def entities(entities):

    st.subheader("🏷 Detected Entities")

    dataframe(entities)