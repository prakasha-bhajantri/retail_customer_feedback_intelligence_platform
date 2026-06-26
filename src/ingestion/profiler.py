"""
profiler.py

Generates dataset statistics for Amazon review data.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean


class DataProfiler:

    def __init__(self):

        self.total_reviews = 0
        self.valid_reviews = 0

        self.rating_counter = Counter()

        self.review_lengths = []

        self.helpful_votes = []

        self.products = Counter()

    def update(self, review: dict):

        self.total_reviews += 1

        self.valid_reviews += 1

        self.rating_counter[review["rating"]] += 1

        self.review_lengths.append(
            len(review["review_text"].split())
        )

        self.helpful_votes.append(
            review["helpful_total"]
        )

        self.products[review["product_id"]] += 1

    def report(self):

        return {

            "total_reviews": self.total_reviews,

            "valid_reviews": self.valid_reviews,

            "average_review_length":

                round(mean(self.review_lengths), 2),

            "average_helpful_votes":

                round(mean(self.helpful_votes), 2),

            "rating_distribution":

                dict(self.rating_counter),

            "top_products":

                self.products.most_common(10)
        }