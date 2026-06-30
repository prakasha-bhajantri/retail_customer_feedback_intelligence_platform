"""
matcher.py

Weak entity matcher for Retail Named Entity Recognition.

Responsibilities
----------------
* Tokenize review text
* Match entities from product metadata
* Generate BIO labels
* Support:
    - Case insensitive matching
    - Partial phrase matching
    - Multiple occurrences
    - Longest match wins
"""

from __future__ import annotations

import re

from typing import List
from typing import Tuple


class RetailEntityMatcher:

    """
    Generates weak BIO labels using
    product metadata.
    """

    ENTITY_PRIORITY = [

        "PRODUCT",

        "SUBCATEGORY",

        "CATEGORY",

        "DEPARTMENT",

    ]

    ####################################################################
    # Tokenization
    ####################################################################

    @staticmethod
    def tokenize(
        text: str,
    ) -> List[str]:

        if text is None:

            return []

        return re.findall(

            r"\w+|[^\w\s]",

            text.lower(),

        )

    ####################################################################
    # Normalize
    ####################################################################

    @staticmethod
    def normalize(
        text,
    ) -> str:

        if text is None:

            return ""

        return str(text).strip().lower()

    ####################################################################
    # Generate Candidate Phrases
    ####################################################################

    @staticmethod
    def generate_candidates(
        phrase: str,
    ):

        phrase = RetailEntityMatcher.normalize(phrase)

        tokens = RetailEntityMatcher.tokenize(phrase)

        candidates = set()

        n = len(tokens)

        for start in range(n):

            for end in range(start + 1, n + 1):

                candidates.add(tuple(tokens[start:end]))

        return sorted(
            [list(c) for c in candidates],
            key=len,
            reverse=True,
        )
    ####################################################################
    # Match
    ####################################################################

    @staticmethod
    def match_entity(

        review_tokens,

        labels,

        entity,

        entity_type,

    ):

        if entity is None:

            return

        candidates = (

            RetailEntityMatcher.generate_candidates(
                entity
            )

        )

        ####################################################
        # Longest first
        ####################################################

        candidates = sorted(

            candidates,

            key=len,

            reverse=True,

        )

        for candidate in candidates:

            n = len(candidate)

            if n == 0:

                continue

            for i in range(

                len(review_tokens)

                -

                n

                +

                1

            ):

                ################################################
                # Already labeled
                ################################################

                if any(

                    label != "O"

                    for label in labels[i:i+n]

                ):

                    continue

                ################################################
                # Phrase Match
                ################################################

                if (

                    review_tokens[i:i+n]

                    ==

                    candidate

                ):

                    labels[i] = (

                        f"B-{entity_type}"

                    )

                    for j in range(

                        1,

                        n,

                    ):

                        labels[i+j] = (

                            f"I-{entity_type}"

                        )

        return labels

    ####################################################################
    # Public API
    ####################################################################

    @classmethod
    def label_review(

        cls,

        review_text,

        product_name,

        department,

        category,

        subcategory,

    ) -> Tuple[List[str], List[str]]:

        tokens = cls.tokenize(

            review_text

        )

        labels = [

            "O"

        ] * len(tokens)

        ####################################################
        # Longest entities first
        ####################################################

        entities = {

            "PRODUCT": product_name,

            "SUBCATEGORY": subcategory,

            "CATEGORY": category,

            "DEPARTMENT": department,

        }

        for entity in cls.ENTITY_PRIORITY:

            cls.match_entity(

                tokens,

                labels,

                entities[entity],

                entity,

            )

        return (

            tokens,

            labels,

        )