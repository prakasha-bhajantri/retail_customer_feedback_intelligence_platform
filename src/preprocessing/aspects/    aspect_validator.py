"""
aspect_validator.py

Validates generated Retail Aspect datasets.

Reports:

- Coverage
- Empty aspect labels
- Aspect frequency
- Invalid labels
- Reviews per aspect
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nlp.aspects.labels import ASPECTS


class AspectDatasetValidator:
    """
    Validates generated aspect datasets.
    """

    def __init__(self):

        self.valid_aspects = set(ASPECTS)

    ####################################################
    # Validate
    ####################################################

    def validate(
        self,
        parquet_path: str | Path,
    ) -> None:

        parquet_path = Path(parquet_path)

        df = pd.read_parquet(parquet_path)

        print("=" * 70)
        print("ASPECT DATASET VALIDATION")
        print("=" * 70)

        print(f"\nDataset : {parquet_path}")
        print(f"Reviews : {len(df):,}")

        ####################################################
        # Coverage
        ####################################################

        has_aspects = (
            df["aspects"]
            .fillna("")
            .str.len() > 0
        )

        coverage = has_aspects.mean()

        print(f"\nCoverage : {coverage:.2%}")

        ####################################################
        # Empty Reviews
        ####################################################

        empty = (~has_aspects).sum()

        print(f"Reviews with no aspects : {empty:,}")

        ####################################################
        # Aspect Counts
        ####################################################

        counts = {}

        invalid = {}

        for aspect_string in df["aspects"].fillna(""):

            if aspect_string == "":
                continue

            for aspect in aspect_string.split(","):

                aspect = aspect.strip()

                if aspect in self.valid_aspects:

                    counts[aspect] = (
                        counts.get(aspect, 0) + 1
                    )

                else:

                    invalid[aspect] = (
                        invalid.get(aspect, 0) + 1
                    )

        print("\nAspect Distribution")
        print("-" * 50)

        aspect_df = (
            pd.DataFrame(
                counts.items(),
                columns=[
                    "Aspect",
                    "Reviews",
                ],
            )
            .sort_values(
                "Reviews",
                ascending=False,
            )
        )

        print(aspect_df.to_string(index=False))

        ####################################################
        # Invalid Labels
        ####################################################

        print("\nInvalid Labels")

        if len(invalid) == 0:

            print("None")

        else:

            invalid_df = (
                pd.DataFrame(
                    invalid.items(),
                    columns=[
                        "Label",
                        "Count",
                    ],
                )
                .sort_values(
                    "Count",
                    ascending=False,
                )
            )

            print(
                invalid_df.to_string(index=False)
            )

        print("\nValidation Complete")