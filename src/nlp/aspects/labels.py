"""
labels.py

Retail Aspect Labels used by the
Retail Customer Feedback Intelligence Platform.

The aspect detector is a multi-label classifier.

One review may mention multiple aspects.
"""

from __future__ import annotations

####################################################
# Retail Aspect Labels
####################################################

ASPECTS = [

    "battery",

    "build_quality",

    "durability",

    "performance",

    "design",

    "appearance",

    "packaging",

    "shipping",

    "delivery",

    "installation",

    "ease_of_use",

    "compatibility",

    "noise",

    "price",

    "value_for_money",

    "customer_service",

    "warranty",

    "returns",

    "availability",

    "accessories",

    "instructions",

    "safety",

    "other",
]

####################################################
# Label Mapping
####################################################

LABEL2ID = {

    label: index

    for index, label in enumerate(
        ASPECTS
    )

}

ID2LABEL = {

    index: label

    for index, label in enumerate(
        ASPECTS
    )

}

####################################################
# Number of Labels
####################################################

NUM_LABELS = len(
    ASPECTS
)