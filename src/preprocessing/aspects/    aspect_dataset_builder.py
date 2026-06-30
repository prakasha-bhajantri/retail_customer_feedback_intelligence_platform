"""
aspect_dataset_builder.py

Automatically generates datasets for
Retail Aspect Detection.

Input
-----
Retail review dataset

Output
------
Retail review dataset with
detected aspect labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.aspects.aspect_matcher import (
    AspectMatcher,
)


class AspectDatasetBuilder:
    """
    Builds aspect datasets using
    dictionary-based matching.
    """

    def __init__(self):

        self.matcher = AspectMatcher()

    ####################################################
    # Build Dataset
    ####################################################

    def build(
        self,
        input_file: str | Path,
        output_file: str | Path,
    ) -> None:

        input_file = Path(input_file)

        output_file = Path(output_file)

        df = pd.read_parquet(
            input_file
        )

        print(
            f"Loaded {len(df):,} reviews."
        )

        ####################################################
        # Detect Aspects
        ####################################################

        df["aspects"] = (

            df["review_text"]

            .fillna("")

            .apply(

                lambda review:

                ",".join(

                    self.matcher.match(review)

                )

            )

        )

        ####################################################
        # Statistics
        ####################################################

        reviews_with_aspects = (

            df["aspects"] != ""

        ).sum()

        print()

        print(
            f"Reviews with aspects : {reviews_with_aspects:,}"
        )

        print(
            f"Coverage             : "
            f"{reviews_with_aspects/len(df):.2%}"
        )

        ####################################################
        # Save
        ####################################################

        output_file.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        df.to_parquet(

            output_file,

            index=False,

        )

        print()

        print(
            f"Saved dataset to:\n{output_file}"
        )

    ####################################################
    # Preview
    ####################################################

    def preview(
        self,
        input_file: str | Path,
        samples: int = 10,
    ) -> pd.DataFrame:

        df = pd.read_parquet(
            input_file
        )

        df["aspects"] = (

            df["review_text"]

            .fillna("")

            .apply(

                lambda review:

                ",".join(

                    self.matcher.match(review)

                )

            )

        )

        return df[
            [
                "review_text",
                "aspects",
            ]
        ].head(samples)