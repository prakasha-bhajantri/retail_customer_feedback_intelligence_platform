"""
Retail Customer Feedback Intelligence Platform

Main Streamlit Application
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
from streamlit_option_menu import option_menu

##############################################################################
# Add Project Root
##############################################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

##############################################################################
# Configuration
##############################################################################

from streamlit_app.config import (
    PAGE_ICON,
    PAGE_TITLE,
    LAYOUT,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

##############################################################################
# Load CSS
##############################################################################

css_path = Path(__file__).parent / "assets" / "style.css"

if css_path.exists():

    with open(css_path) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

##############################################################################
# Import Views
##############################################################################

from src.utils.model_downloader import ModelDownloader

downloader = ModelDownloader()

downloader.download_all_models()

from streamlit_app.views.home import show_home
from streamlit_app.views.review_analysis import show_review_analysis
from streamlit_app.views.dashboard import show_dashboard

##############################################################################
# Sidebar
##############################################################################

with st.sidebar:

    st.image(
        "streamlit_app/assets/logo.png",
        width=110,
    )

    st.markdown(
        "<h3 style='text-align:center;margin-top:-10px;'>Retail Analytics</h3>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Customer Feedback Intelligence Platform"
    )

    st.divider()

    selected = option_menu(

        menu_title=None,

        options=[
            "Home",
            "Review Analysis",
            "Dashboard",
        ],

        icons=[
            "house",
            "search",
            "graph-up",
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "0!important",
                "background-color": "#ffffff",
            },

            "icon": {
                "font-size": "17px",
                "color": "#2563EB",
            },

            "nav-link": {
                "font-size": "14px",
                "font-weight": "500",
                "padding": "10px",
                "margin": "4px 0px",
                "--hover-color": "#EEF4FF",
                "border-radius": "8px",
            },

            "nav-link-selected": {
                "background-color": "#2563EB",
            },

        }

    )

    st.divider()

    st.markdown("### 🤖 Models")

    st.markdown("✅ Sentiment Analysis")

    st.markdown("✅ Retail NER")

    st.markdown("✅ Aspect Engine")

    st.markdown("✅ Business Insights")

    st.divider()

    st.caption("Version 1.0")

##############################################################################
# Routing
##############################################################################

if selected == "Home":

    show_home()

elif selected == "Review Analysis":

    show_review_analysis()

elif selected == "Dashboard":

    show_dashboard()