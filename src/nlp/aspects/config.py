"""
config.py

Configuration for Retail Aspect Detection.

The aspect detector is a multi-label classifier that
predicts which retail aspects are discussed in a review.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AspectConfig:

    ####################################################
    # Dataset
    ####################################################

    TRAIN_DATA = Path(
        "datasets/aspects/train.parquet"
    )

    VALIDATION_DATA = Path(
        "datasets/aspects/validation.parquet"
    )

    TEST_DATA = Path(
        "datasets/aspects/test.parquet"
    )

    ####################################################
    # Model
    ####################################################

    MODEL_NAME = "bert-base-uncased"

    MAX_LENGTH = 256

    ####################################################
    # Training
    ####################################################

    BATCH_SIZE = 32

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
        "artifacts/aspects"
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
        "logs/aspects"
    )

    ####################################################
    # Prediction
    ####################################################

    THRESHOLD = 0.50