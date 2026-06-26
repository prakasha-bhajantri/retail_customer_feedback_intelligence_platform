"""
metadata_joiner.py

Joins review records with product metadata.
"""

from __future__ import annotations


class MetadataJoiner:

    def __init__(self, metadata_records):

        self.catalog = {}

        for product in metadata_records:

            self.catalog[
                product["product_id"]
            ] = product

    def enrich(self, review: dict) -> dict:

        product = self.catalog.get(
            review["product_id"],
            {}
        )

        enriched = review.copy()

        enriched.update({

            "product_name": product.get("product_name"),

            "department": product.get("department"),

            "category": product.get("category"),

            "subcategory": product.get("subcategory"),

            "category_path": product.get("category_path"),

            "image_url": product.get("image_url"),
        })

        return enriched