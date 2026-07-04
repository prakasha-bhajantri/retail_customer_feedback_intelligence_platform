"""
Review Analysis View
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.pipeline.review_pipeline import ReviewPipeline
from streamlit_app.config import SENTIMENT_MODEL

@st.cache_resource
def load_pipeline():
    return ReviewPipeline(
        sentiment_model_path=SENTIMENT_MODEL
    )


pipeline = load_pipeline()


# def show_review_analysis():

#     st.title("📝 Review Analysis")

#     st.caption(
#         "Analyze a single customer review using the NLP pipeline."
#     )

#     st.divider()

#     review = st.text_area(
#         "Customer Review",
#         height=180,
#     )

#     col1, col2 = st.columns(2)

#     with col1:

#         product = st.text_input("Product")

#         department = st.text_input("Department")

#     with col2:

#         category = st.text_input("Category")

#         subcategory = st.text_input("Subcategory")

#     rating = st.slider(
#         "Rating",
#         1,
#         5,
#         5,
#     )

#     if st.button(
#         "Analyze Review",
#         # use_container_width=True,
#         width="stretch",
#     ):

#         if review.strip() == "":

#             st.warning(
#                 "Please enter a review."
#             )

#             return

#         with st.spinner("Analyzing..."):

#             result = pipeline.analyze_review(

#                 review_id="1",

#                 review_text=review,

#                 rating=rating,

#                 product_name=product,

#                 department=department,

#                 category=category,

#                 subcategory=subcategory,
#             )

#         st.success("Analysis Complete")

#         c1, c2 = st.columns(2)

#         c1.metric(
#             "Sentiment",
#             result.sentiment.capitalize(),
#         )

#         c2.metric(
#             "Confidence",
#             f"{result.sentiment_confidence:.2%}",
#         )

#         st.divider()

#         st.subheader("Detected Entities")

#         if result.entities:

#             st.dataframe(

#                 pd.DataFrame(

#                     [
#                         asdict(x)
#                         for x in result.entities
#                     ]

#                 ),

#                 # use_container_width=True,
#                 width="stretch",

#             )

#         else:

#             st.info("No entities detected.")

#         st.divider()

#         st.subheader("Aspect Sentiment")

#         if result.aspects:

#             st.dataframe(

#                 pd.DataFrame(

#                     [
#                         asdict(x)
#                         for x in result.aspects
#                     ]

#                 ),

#                 # use_container_width=True,
#                 width="stretch",

#             )

#         else:

#             st.info("No aspects detected.")

#         st.divider()

#         st.subheader("JSON Output")

#         st.json(

#             {

#                 "review_id": result.review_id,

#                 "sentiment": result.sentiment,

#                 "confidence": result.sentiment_confidence,

#                 "entities": [

#                     asdict(x)

#                     for x in result.entities

#                 ],

#                 "aspects": [

#                     asdict(x)

#                     for x in result.aspects

#                 ],

#             }

#         )

def show_review_analysis():

    st.title("📝 Review Analysis")

    st.caption(
        "Analyze a single customer review using the NLP pipeline."
    )

    st.divider()

    ####################################################
    # Sample Review
    ####################################################

    col1, col2 = st.columns([1, 4])

    with col1:

        if st.button("📄 Load Sample"):

            st.session_state.review = (
                "This cordless drill is amazing. "
                "The battery lasts for hours and the build quality is excellent. "
                "Highly recommended!"
            )

            st.session_state.product = "Cordless Drill"

            st.session_state.department = "Tools"

            st.session_state.category = "Power Tools"

            st.session_state.subcategory = "Drills"

            st.session_state.rating = 5

            st.rerun()

    with col2:

        st.info(
            "New here? Click **Load Sample** to automatically populate the form."
        )

    ####################################################
    # Input Form
    ####################################################

    review = st.text_area(
        "Customer Review",
        height=180,
        key="review",
    )

    col1, col2 = st.columns(2)

    with col1:

        product = st.text_input(
            "Product",
            key="product",
        )

        department = st.text_input(
            "Department",
            key="department",
        )

    with col2:

        category = st.text_input(
            "Category",
            key="category",
        )

        subcategory = st.text_input(
            "Subcategory",
            key="subcategory",
        )

    rating = st.slider(
        "Rating",
        1,
        5,
        key="rating",
    )

    ####################################################
    # Analyze
    ####################################################

    if st.button(
        "Analyze Review",
        width="stretch",
    ):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

            return

        with st.spinner("Analyzing..."):

            result = pipeline.analyze_review(

                review_id="1",

                review_text=review,

                rating=rating,

                product_name=product,

                department=department,

                category=category,

                subcategory=subcategory,
            )

        ####################################################
        # Results
        ####################################################

        st.success("Analysis Complete")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Sentiment",
                result.sentiment.capitalize(),
            )

        with c2:

            st.metric(
                "Confidence",
                f"{result.sentiment_confidence:.2%}",
            )

        ####################################################
        # Entities
        ####################################################

        st.divider()

        st.subheader("🏷 Detected Entities")

        if result.entities:

            st.dataframe(

                pd.DataFrame(

                    [
                        asdict(x)

                        for x in result.entities

                    ]

                ),

                width="stretch",

                hide_index=True,

            )

        else:

            st.info(
                "No entities detected."
            )

        ####################################################
        # Aspects
        ####################################################

        st.divider()

        st.subheader("🎯 Aspect Sentiment")

        if result.aspects:

            st.dataframe(

                pd.DataFrame(

                    [
                        asdict(x)

                        for x in result.aspects

                    ]

                ),

                width="stretch",

                hide_index=True,

            )

        else:

            st.info(
                "No aspects detected."
            )

        ####################################################
        # JSON
        ####################################################

        st.divider()

        st.subheader("📄 JSON Output")

        st.json(

            {

                "review_id": result.review_id,

                "sentiment": result.sentiment,

                "confidence": result.sentiment_confidence,

                "entities": [

                    asdict(x)

                    for x in result.entities

                ],

                "aspects": [

                    asdict(x)

                    for x in result.aspects

                ],

            }

        )

    ####################################################
    # Example
    ####################################################

    st.divider()

    with st.expander("📋 Example Review"):

        st.markdown(
            """
### Example Input

**Customer Review**

> This cordless drill is amazing. The battery lasts for hours and the build quality is excellent. Highly recommended!

| Field | Value |
|-------|-------|
| Product | Cordless Drill |
| Department | Tools |
| Category | Power Tools |
| Subcategory | Drills |
| Rating | ⭐⭐⭐⭐⭐ |

### Expected Output

- 😊 Sentiment: **Positive**
- 🏷 Entity: **Cordless Drill**
- 🎯 Aspect: **Battery**
- 👍 Aspect Sentiment: **Positive**
"""
        )