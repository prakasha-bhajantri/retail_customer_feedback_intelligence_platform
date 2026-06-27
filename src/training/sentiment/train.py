"""
train.py

Entry point for Retail Sentiment Model Training.

Responsibilities
----------------
* Configure logging
* Set random seed
* Load tokenizer
* Build datasets
* Create dataloaders
* Create model
* Configure optimizer & scheduler
* Train model
* Evaluate model
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from src.training.sentiment.class_weights import (
    ClassWeightCalculator,
)
from src.training.sentiment.config import (
    SentimentConfig,
)
from src.training.sentiment.dataloader import (
    RetailDataLoaderFactory,
)
from src.training.sentiment.dataset import (
    RetailSentimentDataset,
)
from src.training.device import (
    get_device,
)
from src.training.sentiment.losses import (
    LossFactory,
)
from src.training.sentiment.model_factory import (
    ModelFactory,
)
from src.training.sentiment.optimizer import (
    OptimizerFactory,
)
from src.training.sentiment.scheduler import (
    SchedulerFactory,
)
from src.training.sentiment.trainer import (
    TransformerTrainer,
)

logger = logging.getLogger(__name__)

def configure_logging() -> None:
    """
    Configure application logging.
    """

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(message)s",
    # )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_device():
    """
    Select the best available device.
    """

    device = get_device()

    logger.info(
        "Using device: %s",
        device,
    )

    return device

def create_tokenizer(config):
    """
    Load Hugging Face tokenizer.
    """

    logger.info(
        "Loading tokenizer: %s",
        config.MODEL_NAME,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
    )

    return tokenizer

def create_datasets(
    tokenizer,
    config
):
    """
    Create train, validation and test datasets.
    """

    logger.info("Loading train/validation/test datasets...")
    

    train_dataset = RetailSentimentDataset(
        parquet_path=config.TRAIN_DATA,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    validation_dataset = RetailSentimentDataset(
        parquet_path=config.VALIDATION_DATA,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    test_dataset = RetailSentimentDataset(
        parquet_path=config.TEST_DATA,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    logger.info(
        f"Train Samples      : {len(train_dataset):,}"
    )

    logger.info(
        f"Validation Samples : {len(validation_dataset):,}"
    )

    logger.info(
        f"Test Samples       : {len(test_dataset):,}"
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )

def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    config
):
    """
    Create PyTorch DataLoaders.
    """

    logger.info("Creating dataloaders...")

    train_loader = RetailDataLoaderFactory.create(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    validation_loader = RetailDataLoaderFactory.create(
        dataset=validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = RetailDataLoaderFactory.create(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    logger.info(
        f"Train Batches      : {len(train_loader):,}"
    )

    logger.info(
        f"Validation Batches : {len(validation_loader):,}"
    )

    logger.info(
        f"Test Batches       : {len(test_loader):,}"
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )

def create_class_weights(
    device,
    config
):
    """
    Compute balanced class weights.
    """

    logger.info(
        "Computing class weights..."
    )

    class_weights = (
        ClassWeightCalculator.compute_tensor(
            parquet_path=config.TRAIN_DATA,
            device=device,
        )
    )

    logger.info(
        "Class Weights : %s",
        class_weights.tolist(),
    )

    return class_weights

def create_loss(
    class_weights,
):
    """
    Create loss function.
    """

    logger.info(
        "Creating loss function..."
    )

    return LossFactory.cross_entropy(
        class_weights=class_weights,
    )

def create_model(
    device,
    config
):
    """
    Create transformer model.
    """

    logger.info(
        "Loading model : %s",
        config.MODEL_NAME,
    )

    model = ModelFactory.create(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    model.to(device)

    logger.info(
        "Model moved to %s",
        device,
    )

    return model

def create_optimizer(
    model,
    config
):
    """
    Create AdamW optimizer.
    """

    logger.info(
        "Creating optimizer..."
    )

    return OptimizerFactory.adamw(
        model=model,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

def create_scheduler(
    optimizer,
    train_loader,
    config
):
    """
    Create linear warmup scheduler.
    """

    logger.info(
        "Creating learning rate scheduler..."
    )

    total_training_steps = (

        len(train_loader)

        *

        config.NUM_EPOCHS
    )

    scheduler = SchedulerFactory.linear_warmup(
        optimizer=optimizer,
        total_training_steps=total_training_steps,
        warmup_ratio=config.WARMUP_RATIO,
    )

    logger.info(
        "Total Training Steps : %d",
        total_training_steps,
    )

    return scheduler

def create_trainer(
    model,
    optimizer,
    scheduler,
    criterion,
    tokenizer,
    device,
    config
):
    """
    Create transformer trainer.
    """

    logger.info(
        "Initializing trainer..."
    )

    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        gradient_clip=config.GRADIENT_CLIP,
        checkpoint_dir=config.CHECKPOINT_DIR,
    )

    return trainer

def main(
    resume: bool = False,
    config=SentimentConfig
):
    
    """
    Main training entry point.
    """

    ####################################################
    # Logging
    ####################################################

    configure_logging()

    ####################################################
    # Random Seed
    ####################################################

    set_random_seed(
        config.RANDOM_SEED
    )

    ####################################################
    # Device
    ####################################################

    device = create_device()

    ####################################################
    # Tokenizer
    ####################################################

    tokenizer = create_tokenizer(config)

    ####################################################
    # Dataset
    ####################################################

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        tokenizer,
        config
    )

    ####################################################
    # DataLoaders
    ####################################################

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        config
    )

    ####################################################
    # Class Weights
    ####################################################

    class_weights = create_class_weights(
        device,
        config
    )

    ####################################################
    # Loss
    ####################################################

    criterion = create_loss(
        class_weights,
    )

    ####################################################
    # Model
    ####################################################

    if resume:

        logger.info(
                "Loading checkpoint from %s",
                config.LAST_MODEL,
        )

        model = ModelFactory.load(
            config.LAST_MODEL,
        )

        model.to(device)

    else:

        model = create_model(
            device,
            config
        )

    ####################################################
    # Optimizer
    ####################################################

    optimizer = create_optimizer(
        model,
        config
    )

    ####################################################
    # Scheduler
    ####################################################

    scheduler = create_scheduler(
        optimizer,
        train_loader,
        config
    )

    ####################################################
    # Trainer
    ####################################################

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        config=config
    )

    ####################################################
    # Resume Training State
    ####################################################

    if resume:
        logger.info(

            "Restoring optimizer and scheduler state..."

        )
        trainer.load_checkpoint(
            config.LAST_MODEL
        )

    ####################################################
    # Train
    ####################################################
    logger.info(
    "Starting model training..."
    )

    trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=config.NUM_EPOCHS,
    )

    ####################################################
    # Test Evaluation
    ####################################################

    print()

    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    trainer.evaluate(
        test_loader,
    )

    logger.info(
    "Training and evaluation completed successfully."
    )

if __name__ == "__main__":

    main(resume=False)