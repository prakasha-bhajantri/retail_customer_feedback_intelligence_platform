"""
build_dataset.py

Builds Retail Aspect datasets for
train, validation and test splits.

Pipeline

Raw Dataset
      │
      ▼
Aspect Matching
      │
      ▼
Dataset Generation
      │
      ▼
Validation
      │
      ▼
Parquet Output
"""

from pathlib import Path

from src.preprocessing.aspects.aspect_dataset_builder import (
    AspectDatasetBuilder,
)
from src.preprocessing.aspects.aspect_validator import (
    AspectDatasetValidator,
)


TRAIN_INPUT = Path(
    "datasets/splits/train.parquet"
)

VALIDATION_INPUT = Path(
    "datasets/splits/validation.parquet"
)

TEST_INPUT = Path(
    "datasets/splits/test.parquet"
)

OUTPUT_DIR = Path(
    "datasets/aspects"
)


def build():

    builder = AspectDatasetBuilder()

    validator = AspectDatasetValidator()

    datasets = [

        (
            TRAIN_INPUT,
            OUTPUT_DIR / "train.parquet",
        ),

        (
            VALIDATION_INPUT,
            OUTPUT_DIR / "validation.parquet",
        ),

        (
            TEST_INPUT,
            OUTPUT_DIR / "test.parquet",
        ),
    ]

    for input_file, output_file in datasets:

        print()

        print("=" * 80)

        print(f"Processing {input_file.name}")

        print("=" * 80)

        builder.build(

            input_file=input_file,

            output_file=output_file,

        )

        print()

        validator.validate(
            output_file
        )

    print()

    print("=" * 80)

    print("Aspect Dataset Generation Completed")

    print("=" * 80)


if __name__ == "__main__":

    build()