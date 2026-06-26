"""
retail_data_pipeline.py

End-to-end data pipeline for building the Gold Retail Dataset.
"""

from pathlib import Path
import pandas as pd

from src.ingestion.loader import AmazonReviewLoader
from src.ingestion.meta_data_loader import MetadataLoader

from src.ingestion.mapper import AmazonReviewMapper
from src.ingestion.metadata_mapper import MetadataMapper

from src.ingestion.validator import RetailReviewValidator
from src.ingestion.metadata_joiner import MetadataJoiner

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.sentiment_mapper import SentimentMapper
from src.preprocessing.feature_engineering import FeatureEngineer


class RetailDataPipeline:

    def __init__(
        self,
        review_path: str,
        metadata_path: str,
        output_path: str,
    ):
        self.review_path = Path(review_path)
        self.metadata_path = Path(metadata_path)
        self.output_path = Path(output_path)

    def load_catalog(self) -> MetadataJoiner:
        """
        Load product metadata and create an in-memory lookup catalog.
        """

        loader = MetadataLoader(self.metadata_path)

        products = []

        for record in loader.stream():
            mapped = MetadataMapper.map(record)
            products.append(mapped)

        return MetadataJoiner(products)

    def build(self) -> pd.DataFrame:
        """
        Execute the complete data pipeline and return
        the processed DataFrame.
        """

        print("Loading product catalog...")
        joiner = self.load_catalog()

        print("Loading reviews...")
        loader = AmazonReviewLoader(self.review_path)

        processed_reviews = []

        total = 0
        valid = 0
        skipped = 0

        for record in loader.stream():

            total += 1

            # -------------------------------
            # Map raw review
            # -------------------------------
            review = AmazonReviewMapper.map(record)

            # -------------------------------
            # Validate
            # -------------------------------
            is_valid, _ = RetailReviewValidator.validate(review)

            if not is_valid:
                skipped += 1
                continue

            valid += 1

            # -------------------------------
            # Clean review text
            # -------------------------------
            review["review_text"] = TextCleaner.clean(
                review["review_text"]
            )

            # -------------------------------
            # Join product metadata
            # -------------------------------
            review = joiner.enrich(review)

            # -------------------------------
            # Sentiment label
            # -------------------------------
            review["sentiment_label"] = (
                SentimentMapper.map(review["rating"])
            )

            # -------------------------------
            # Feature Engineering
            # -------------------------------
            review = FeatureEngineer.generate(review)

            processed_reviews.append(review)

        print(f"Total Reviews : {total}")
        print(f"Valid Reviews : {valid}")
        print(f"Skipped       : {skipped}")

        df = pd.DataFrame(processed_reviews)

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save Gold Dataset
        df.to_parquet(
            self.output_path,
            index=False
        )

        print(f"\nGold Dataset saved to:\n{self.output_path}")
        print(f"Shape: {df.shape}")

        return df