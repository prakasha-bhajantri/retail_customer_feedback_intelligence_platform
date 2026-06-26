"""
mapper.py

Converts raw Amazon review records into the project's
canonical retail review schema.
"""

from __future__ import annotations

from typing import Dict
from uuid import uuid4

class AmazonReviewMapper:
    """
    Maps Amazon review schema to internal retail schema.
    """

    SOURCE = "amazon"

    @staticmethod
    def map(record: Dict) -> Dict:

        helpful = record.get("helpful", [0, 0])

        return {

            # Internal identifiers
            "review_id": str(uuid4()),

            "source": AmazonReviewMapper.SOURCE,

            # Product
            "product_id": record.get("asin"),

            # Customer
            "customer_id": record.get("reviewerID"),

            "customer_name": record.get("reviewerName"),

            # Review
            "review_title": record.get("summary"),

            "review_text": record.get("reviewText"),

            # Rating
            "rating": float(record.get("overall", 0)),

            # Helpfulness
            "helpful_yes": helpful[0],

            "helpful_total": helpful[1],

            # Time
            "review_timestamp": record.get("unixReviewTime"),

            "review_date": record.get("reviewTime"),
            
            # To be populated later

            "sentiment_label": None,

            "language": None,
        }
    
    