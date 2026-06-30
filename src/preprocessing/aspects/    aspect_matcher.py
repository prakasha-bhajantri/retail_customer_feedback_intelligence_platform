"""
aspect_matcher.py

Dictionary-based Retail Aspect Matcher.

Uses keyword matching to identify
retail aspects mentioned in customer reviews.
"""

from __future__ import annotations

import re
from typing import List, Set

from src.preprocessing.aspects.aspect_dictionary import (
    ASPECT_DICTIONARY,
)


class AspectMatcher:
    """
    Matches retail aspects using
    dictionary-based keyword lookup.
    """

    def __init__(self):

        self.aspect_dictionary = ASPECT_DICTIONARY

    ####################################################
    # Text Cleaning
    ####################################################

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Normalize review text before matching.
        """

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    ####################################################
    # Match One Review
    ####################################################

    def match(
        self,
        review: str,
    ) -> List[str]:
        """
        Returns all matched retail aspects.
        """

        review = self.normalize_text(
            review
        )

        matched: Set[str] = set()

        for aspect, keywords in self.aspect_dictionary.items():

            for keyword in keywords:

                pattern = (
                    r"\b"
                    + re.escape(keyword.lower())
                    + r"\b"
                )

                if re.search(
                    pattern,
                    review,
                ):

                    matched.add(
                        aspect
                    )

                    break

        return sorted(
            list(matched)
        )

    ####################################################
    # Match Multiple Reviews
    ####################################################

    def match_batch(
        self,
        reviews: List[str],
    ) -> List[List[str]]:
        """
        Detect aspects for multiple reviews.
        """

        return [

            self.match(review)

            for review in reviews

        ]