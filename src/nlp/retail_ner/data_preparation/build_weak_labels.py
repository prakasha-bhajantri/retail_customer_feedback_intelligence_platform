"""
build_weak_labels.py

Automatically generates weak BIO labels for Retail NER.

Input
-----
master_reviews.parquet

Output
------
ner_train.parquet

Entities
--------
PRODUCT
DEPARTMENT
CATEGORY
SUBCATEGORY
"""

from __future__ import annotations

import re

from pathlib import Path

import pandas as pd

from tqdm.auto import tqdm


LABELS = {
    "O": 0,

    "B-PRODUCT": 1,
    "I-PRODUCT": 2,

    "B-DEPARTMENT": 3,
    "I-DEPARTMENT": 4,

    "B-CATEGORY": 5,
    "I-CATEGORY": 6,

    "B-SUBCATEGORY": 7,
    "I-SUBCATEGORY": 8,
}


class WeakLabelBuilder:

    def __init__(
        self,
        input_file: str | Path,
        output_file: str | Path,
    ):

        self.input_file = Path(input_file)

        self.output_file = Path(output_file)

    ###############################################################

    @staticmethod
    def tokenize(
        text: str,
    ):

        return re.findall(
            r"\w+|[^\w\s]",
            text.lower(),
        )

    ###############################################################

    @staticmethod
    def label_entity(
        tokens,
        labels,
        entity,
        prefix,
    ):

        if (
            entity is None
            or
            pd.isna(entity)
        ):
            return labels

        entity_tokens = re.findall(
            r"\w+|[^\w\s]",
            str(entity).lower(),
        )

        if len(entity_tokens) == 0:
            return labels

        n = len(entity_tokens)

        for i in range(

            len(tokens) - n + 1

        ):

            if tokens[i:i+n] == entity_tokens:

                labels[i] = f"B-{prefix}"

                for j in range(
                    1,
                    n,
                ):
                    labels[i+j] = f"I-{prefix}"

        return labels

    ###############################################################

    def process(self):

        df = pd.read_parquet(
            self.input_file
        )

        records = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
        ):

            tokens = self.tokenize(
                row["review_text"]
            )

            labels = [

                "O"

            ] * len(tokens)

            labels = self.label_entity(
                tokens,
                labels,
                row["product_name"],
                "PRODUCT",
            )

            labels = self.label_entity(
                tokens,
                labels,
                row["department"],
                "DEPARTMENT",
            )

            labels = self.label_entity(
                tokens,
                labels,
                row["category"],
                "CATEGORY",
            )

            labels = self.label_entity(
                tokens,
                labels,
                row["subcategory"],
                "SUBCATEGORY",
            )

            label_ids = [

                LABELS[x]

                for x in labels

            ]

            records.append({

                "review_id":
                    row["review_id"],

                "tokens":
                    tokens,

                "ner_tags":
                    label_ids,

            })

        output = pd.DataFrame(
            records
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.to_parquet(
            self.output_file,
            index=False,
        )

        print()

        print("=" * 60)

        print(
            f"Saved {len(output):,} records"
        )

        print(
            self.output_file
        )

        print("=" * 60)


if __name__ == "__main__":

    builder = WeakLabelBuilder(

        input_file="datasets/splits/train.parquet",

        output_file="datasets/ner/ner_train.parquet",

    )

    builder.process()