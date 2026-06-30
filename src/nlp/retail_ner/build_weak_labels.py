"""
build_weak_labels.py

Generate weakly labeled BIO datasets for Retail NER.

Input
-----
train.parquet

Output
------
ner_train.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.nlp.retail_ner.matcher import RetailEntityMatcher

logger = logging.getLogger(__name__)


class WeakLabelBuilder:
    """
    Generates weak BIO labels for Retail NER.
    """

    REQUIRED_COLUMNS = [
        "review_id",
        "review_text",
        "product_name",
        "department",
        "category",
        "subcategory",
    ]

    def __init__(
        self,
        input_file: str | Path,
        output_file: str | Path,
    ):

        self.input_file = Path(input_file)

        self.output_file = Path(output_file)

    ###########################################################

    def load(self) -> pd.DataFrame:

        logger.info(
            "Loading dataset..."
        )

        df = pd.read_parquet(
            self.input_file
        )

        missing = [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns: {missing}"

            )

        logger.info(
            "Loaded %s reviews",
            f"{len(df):,}",
        )

        return df

    ###########################################################

    def build(self):

        df = self.load()

        output = []

        logger.info(
            "Generating BIO labels..."
        )

        for row in tqdm(

            df.itertuples(index=False),

            total=len(df),

        ):

            tokens, labels = (

                RetailEntityMatcher.label_review(

                    review_text=row.review_text,

                    product_name=row.product_name,

                    department=row.department,

                    category=row.category,

                    subcategory=row.subcategory,

                )

            )

            output.append(

                {

                    "review_id": row.review_id,

                    "tokens": tokens,

                    "ner_tags": labels,

                }

            )

        output = pd.DataFrame(
            output
        )

        self.output_file.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        output.to_parquet(

            self.output_file,

            index=False,

        )

        logger.info(
            "Saved %s weakly labeled reviews",
            f"{len(output):,}",
        )

        logger.info(

            "Output: %s",

            self.output_file,

        )


###############################################################

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    builder = WeakLabelBuilder(

        input_file="datasets/splits/train.parquet",

        output_file="datasets/ner/ner_train.parquet",

    )

    builder.build()