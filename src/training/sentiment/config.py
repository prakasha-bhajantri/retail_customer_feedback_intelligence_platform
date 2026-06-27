"""
config.py

Configuration for Retail Sentiment Model Training.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SentimentConfig:

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    # TRAIN_DATA = Path("datasets/splits/train.parquet")
    # VALIDATION_DATA = Path("datasets/splits/validation.parquet")
    # TEST_DATA = Path("datasets/splits/test.parquet")

    TRAIN_DATA = Path("datasets/splits/train_small.parquet")
    VALIDATION_DATA = Path("datasets/splits/validation_small.parquet")
    TEST_DATA = Path("datasets/splits/test_small.parquet")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    MODEL_NAME = "bert-base-uncased"

    NUM_CLASSES = 3

    MAX_LENGTH = 256

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    BATCH_SIZE = 32

    NUM_EPOCHS = 3

    LEARNING_RATE = 2e-5

    WEIGHT_DECAY = 0.01

    WARMUP_RATIO = 0.10

    GRADIENT_CLIP = 1.0

    RANDOM_SEED = 42

    # ---------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------

    NUM_WORKERS = 4

    # Local Mac
    PIN_MEMORY = False

    # Vertex AI GPU
    # PIN_MEMORY = True    

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------

    CHECKPOINT_DIR = Path("artifacts/sentiment")

    BEST_MODEL = CHECKPOINT_DIR / "best_model"

    LAST_MODEL = CHECKPOINT_DIR / "last_model"

    LOG_DIR = Path("logs/sentiment")

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    LABEL2ID = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    ID2LABEL = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }