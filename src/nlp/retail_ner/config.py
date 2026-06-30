"""
config.py

Configuration for Retail Named Entity Recognition.

Uses Hugging Face token classification models.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetailNERConfig:

    ####################################################
    # Dataset
    ####################################################

    TRAIN_DATA = Path("datasets/ner/ner_train_small.parquet")

    VALIDATION_DATA = Path("datasets/ner/ner_validation_small.parquet")

    TEST_DATA = Path("datasets/ner/ner_test_small.parquet")

    ####################################################
    # Model
    ####################################################

    MODEL_NAME = "bert-base-uncased"

    MAX_LENGTH = 256

    ####################################################
    # Training
    ####################################################

    BATCH_SIZE = 16

    NUM_EPOCHS = 5

    LEARNING_RATE = 2e-5

    WEIGHT_DECAY = 0.01

    WARMUP_RATIO = 0.10

    GRADIENT_CLIP = 1.0

    RANDOM_SEED = 42

    ####################################################
    # Hardware
    ####################################################

    NUM_WORKERS = 4

    PIN_MEMORY = True

    ####################################################
    # Saving
    ####################################################

    CHECKPOINT_DIR = Path(
        "artifacts/retail_ner"
    )

    BEST_MODEL = (
        CHECKPOINT_DIR /
        "best_model"
    )

    LAST_MODEL = (
        CHECKPOINT_DIR /
        "last_model"
    )

    LOG_DIR = Path(
        "logs/retail_ner"
    )