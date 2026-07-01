"""
Configuration for Streamlit Application.
"""

from pathlib import Path

####################################################
# Project
####################################################

PROJECT_NAME = "Retail Customer Feedback Intelligence Platform"

PAGE_TITLE = PROJECT_NAME

PAGE_ICON = "🛍️"

LAYOUT = "wide"

####################################################
# Models
####################################################

SENTIMENT_MODEL = Path(
    "artifacts/sentiment/best_model"
)

####################################################
# Upload
####################################################

SUPPORTED_FILE_TYPES = [
    "csv",
]

####################################################
# Theme
####################################################

PRIMARY_COLOR = "#2563EB"

POSITIVE_COLOR = "#16A34A"

NEGATIVE_COLOR = "#DC2626"

NEUTRAL_COLOR = "#F59E0B"