"""
data_splitter.py

Creates train, validation and test datasets
for sentiment model training.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Splits the Gold Dataset into
    Train / Validation / Test datasets.
    """

    def __init__(self, parquet_path: str):

        self.parquet_path = Path(parquet_path)

        self.df = pd.read_parquet(self.parquet_path)

    def print_distribution(
        self,
        df: pd.DataFrame,
        name: str,
    ) -> None:
        """
        Prints sentiment class distribution.
        """

        print(f"\n{name}")

        distribution = (
            df["sentiment_label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print(distribution)

    def split(
        self,
        output_dir: str,
        train_size: float = 0.8,
        validation_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ):

        assert (
            abs(train_size + validation_size + test_size - 1.0)
            < 1e-6
        ), "Train + Validation + Test must equal 1."

        print("=" * 70)
        print("Retail Customer Feedback Dataset Splitter")
        print("=" * 70)

        print(f"\nOriginal Reviews : {len(self.df):,}")

        # --------------------------------------------------
        # Remove duplicate review text
        # --------------------------------------------------

        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=["review_text"]
        ).reset_index(drop=True)

        duplicates_removed = before - len(self.df)

        print(f"Duplicates Removed : {duplicates_removed:,}")
        print(f"Remaining Reviews  : {len(self.df):,}")

        # --------------------------------------------------
        # Train / Temp Split
        # --------------------------------------------------

        train_df, temp_df = train_test_split(
            self.df,
            test_size=(1 - train_size),
            stratify=self.df["sentiment_label"],
            random_state=random_state,
        )

        # --------------------------------------------------
        # Validation / Test Split
        # --------------------------------------------------

        validation_ratio = validation_size / (
            validation_size + test_size
        )

        validation_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - validation_ratio),
            stratify=temp_df["sentiment_label"],
            random_state=random_state,
        )

        # --------------------------------------------------
        # Save datasets
        # --------------------------------------------------

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df.to_parquet(
            output_dir / "train.parquet",
            index=False,
        )

        validation_df.to_parquet(
            output_dir / "validation.parquet",
            index=False,
        )

        test_df.to_parquet(
            output_dir / "test.parquet",
            index=False,
        )

        # --------------------------------------------------
        # Dataset Summary
        # --------------------------------------------------

        total = len(self.df)

        print("\n" + "=" * 70)
        print("Dataset Summary")
        print("=" * 70)

        print(
            f"Train       : {len(train_df):,} "
            f"({len(train_df)/total:.1%})"
        )

        print(
            f"Validation  : {len(validation_df):,} "
            f"({len(validation_df)/total:.1%})"
        )

        print(
            f"Test        : {len(test_df):,} "
            f"({len(test_df)/total:.1%})"
        )

        # --------------------------------------------------
        # Verify Class Distribution
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("Sentiment Distribution")
        print("=" * 70)

        self.print_distribution(train_df, "TRAIN")

        self.print_distribution(validation_df, "VALIDATION")

        self.print_distribution(test_df, "TEST")

        print("\nDataset splitting completed successfully.")

        return (
            train_df,
            validation_df,
            test_df,
        )