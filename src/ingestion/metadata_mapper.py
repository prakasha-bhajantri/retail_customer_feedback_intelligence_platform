"""
metadata_mapper.py

Maps Amazon product metadata into the project's
canonical product schema.
"""

from __future__ import annotations

from typing import Dict


class MetadataMapper:

    SOURCE = "amazon"

    @staticmethod
    def map(record: Dict) -> Dict:

        categories = record.get("categories", [])

        department = None
        category = None
        subcategory = None
        category_path = None

        if categories:

            path = categories[0]

            category_path = " > ".join(path)

            if len(path) > 1:
                department = path[1]

            if len(path) > 2:
                category = path[2]

            if len(path) > 3:
                subcategory = path[-1]

        return {

            "product_id": record.get("asin"),

            "product_name": record.get("title"),

            "department": department,

            "category": category,

            "subcategory": subcategory,

            "category_path": category_path,

            "image_url": record.get("imUrl"),
        }