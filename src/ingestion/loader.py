"""
loader.py

Streams Amazon review data without loading the entire dataset into memory.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Generator


# class AmazonReviewLoader:
#     """
#     Streams Amazon review records.

#     Example
#     -------
#     loader = AmazonReviewLoader("reviews.json.gz")

#     for review in loader.stream():
#         print(review)
#     """

#     def __init__(self, file_path: str | Path):

#         self.file_path = Path(file_path)

#     def stream(self) -> Generator[Dict, None, None]:

#         with gzip.open(self.file_path, "rt", encoding="utf-8") as file:

#             for line in file:

#                 if not line.strip():
#                     continue

#                 yield json.loads(line)

from src.ingestion.base_loader import BaseLoader

class AmazonReviewLoader(BaseLoader):

    def __init__(self, file_path):

        super().__init__(
            file_path=file_path,
            parser="json"
        )