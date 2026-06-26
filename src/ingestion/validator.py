from __future__ import annotations

import re
from typing import Dict, Tuple


class RetailReviewValidator:

    @staticmethod
    def validate(review: Dict) -> Tuple[bool, str]:

        if not review["product_id"]:
            return False, "Missing product_id"

        if not review["review_text"]:
            return False, "Missing review_text"

        text = review["review_text"].strip()

        if len(text) < 5:
            return False, "Review too short"

        if review["rating"] not in [1, 2, 3, 4, 5]:
            return False, "Invalid rating"

        if not re.search(r"[A-Za-z]", text):
            return False, "No valid text"

        return True, "Valid"