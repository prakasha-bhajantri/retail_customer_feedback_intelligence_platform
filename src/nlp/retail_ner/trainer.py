"""
trainer.py

Enterprise Trainer for Retail Named Entity Recognition.

Supports

- BERT
- RoBERTa
- DeBERTa
- DistilBERT
- ModernBERT

Features
--------
* Training
* Validation
* Prediction
* Batch Prediction
* Early Stopping
* Resume Training
* Checkpoint Saving
* History Tracking
* Gradient Clipping
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

import torch

from tqdm.auto import tqdm

from src.nlp.retail_ner.metrics import (
    NERMetrics,
)

logger = logging.getLogger(__name__)

class NERTrainer:
    """
    Trainer for Hugging Face
    Token Classification models.
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        tokenizer=None,
        checkpoint_dir="artifacts/retail_ner",
        gradient_clip=1.0,
        patience=3,
        start_epoch=1,
    ):
        self.model = model

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        self.device = device

        self.tokenizer = tokenizer

        self.gradient_clip = gradient_clip

        self.patience = patience

        self.start_epoch = start_epoch

        self.checkpoint_dir = Path(checkpoint_dir)

        self.best_model_dir = (
            self.checkpoint_dir / "best_model"
        )

        self.last_model_dir = (
            self.checkpoint_dir / "last_model"
        )

        self.report_dir = (
            self.checkpoint_dir / "reports"
        )

        for directory in (
            self.best_model_dir,
            self.last_model_dir,
            self.report_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )        

        self.best_f1 = -1.0

        self.best_epoch = -1

        self.current_epoch = 0

        self.early_stop_counter = 0

        self.history = {

                "epoch": [],

                "train_loss": [],

                "validation_loss": [],

                "accuracy": [],

                "precision": [],

                "recall": [],

                "f1": [],

                "learning_rate": [],
            }
        
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]
####################################################################
# Training
####################################################################

    def train_epoch(
        self,
        train_loader,
    ) -> float:

        self.model.train()

        running_loss = 0.0

        progress = tqdm(

            train_loader,

            desc=f"Epoch {self.current_epoch} Training",

            leave=False,
        )

        for batch in progress:

            input_ids = batch[
                "input_ids"
            ].to(self.device)

            attention_mask = batch[
                "attention_mask"
            ].to(self.device)

            labels = batch[
                "labels"
            ].to(self.device)

            ####################################################
            # Forward
            ####################################################

            outputs = self.model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels,
            )

            loss = outputs.loss

            ####################################################
            # Backward
            ####################################################

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            ####################################################
            # Gradient Clipping
            ####################################################

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip,
            )

            ####################################################
            # Optimizer Step
            ####################################################

            self.optimizer.step()

            if self.scheduler is not None:

                self.scheduler.step()

            ####################################################
            # Statistics
            ####################################################

            running_loss += loss.item()

            progress.set_postfix(

                loss=f"{loss.item():.4f}",

                lr=f"{self.current_lr():.2e}",
            )

        epoch_loss = (

            running_loss /

            len(train_loader)
        )

        logger.info(

            "Training Loss : %.4f",

            epoch_loss,
        )

        return epoch_loss    
    
    ####################################################################
    # Validation
    ####################################################################

    def validate_epoch(
        self,
        validation_loader,
    ):

        self.model.eval()

        running_loss = 0.0

        predictions = []

        labels_list = []

        progress = tqdm(

            validation_loader,

            desc=f"Epoch {self.current_epoch} Validation",

            leave=False,
        )

        with torch.no_grad():

            for batch in progress:

                input_ids = batch[
                    "input_ids"
                ].to(self.device)

                attention_mask = batch[
                    "attention_mask"
                ].to(self.device)

                labels = batch[
                    "labels"
                ].to(self.device)

                ####################################################
                # Forward
                ####################################################

                outputs = self.model(

                    input_ids=input_ids,

                    attention_mask=attention_mask,

                    labels=labels,
                )

                loss = outputs.loss

                logits = outputs.logits

                running_loss += loss.item()

                ####################################################
                # Predictions
                ####################################################

                preds = torch.argmax(

                    logits,

                    dim=-1,

                )

                predictions.extend(

                    preds.cpu().numpy()

                )

                labels_list.extend(

                    labels.cpu().numpy()

                )

                progress.set_postfix(

                    loss=f"{loss.item():.4f}"

                )

        ####################################################
        # Validation Loss
        ####################################################

        validation_loss = (

            running_loss /

            len(validation_loader)

        )

        ####################################################
        # Metrics
        ####################################################

        metrics = NERMetrics.compute(

            labels_list,

            predictions,

        )

        report = (

            NERMetrics.classification_report(

                labels_list,

                predictions,

            )

        )

        ####################################################
        # Save Report
        ####################################################

        report_file = (

            self.report_dir /

            "classification_report.txt"

        )

        with open(

            report_file,

            "w",

            encoding="utf-8",

        ) as file:

            file.write(report)

        ####################################################
        # Logging
        ####################################################

        logger.info(

            "Validation Loss : %.4f",

            validation_loss,

        )

        logger.info(

            "Validation Entity F1 : %.4f",

            metrics["f1"],

        )

        return (

            validation_loss,

            metrics,

        )
        
    ####################################################################
    # Training Loop
    ####################################################################

    def fit(
        self,
        train_loader,
        validation_loader,
        epochs: int,
    ):

        logger.info(
            "Starting Training..."
        )

        for epoch in range(
            self.start_epoch,
            epochs + 1,
        ):

            self.current_epoch = epoch

            current_lr = self.current_lr()

            print()

            print("=" * 70)

            print(
                f"Epoch {epoch}/{epochs}"
            )

            print("=" * 70)

            ####################################################
            # Train
            ####################################################

            train_loss = self.train_epoch(
                train_loader
            )

            ####################################################
            # Validate
            ####################################################

            validation_loss, metrics = (

                self.validate_epoch(
                    validation_loader
                )

            )

            ####################################################
            # History
            ####################################################

            self.history["epoch"].append(epoch)

            self.history["train_loss"].append(
                train_loss
            )

            self.history[
                "validation_loss"
            ].append(
                validation_loss
            )

            self.history["accuracy"].append(
                metrics["accuracy"]
            )

            self.history["precision"].append(
                metrics["precision"]
            )

            self.history["recall"].append(
                metrics["recall"]
            )

            self.history["f1"].append(
                metrics["f1"]
            )

            self.history[
                "learning_rate"
            ].append(
                current_lr
            )

            ####################################################
            # Summary
            ####################################################

            print()

            print(
                f"Train Loss      : {train_loss:.4f}"
            )

            print(
                f"Validation Loss : {validation_loss:.4f}"
            )

            print(
                f"Accuracy         : {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision        : {metrics['precision']:.4f}"
            )

            print(
                f"Recall           : {metrics['recall']:.4f}"
            )

            print(
                f"Entity F1        : {metrics['f1']:.4f}"
            )

            print(
                f"Learning Rate    : {current_lr:.2e}"
            )

            ####################################################
            # Best Model
            ####################################################

            if metrics["f1"] > self.best_f1:

                logger.info(
                    "New Best Model Found."
                )

                self.best_f1 = metrics["f1"]

                self.best_epoch = epoch

                self.early_stop_counter = 0

                self.save_checkpoint(

                    self.best_model_dir,

                    epoch,

                )

            else:

                self.early_stop_counter += 1

            ####################################################
            # Save Last Model
            ####################################################

            self.save_checkpoint(

                self.last_model_dir,

                epoch,

            )

            ####################################################
            # Early Stopping
            ####################################################

            if (
                self.early_stop_counter
                >= self.patience
            ):

                logger.info(
                    "Early stopping triggered."
                )

                break

        ####################################################
        # Save History
        ####################################################

        self.save_history()

        logger.info(
            "Training Completed."
        )    
    def save_checkpoint(
        self,
        checkpoint_dir,
        epoch,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)

        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "best_f1": self.best_f1,
                "best_epoch": self.best_epoch,
            },
            checkpoint_dir / "training_state.pt",
        )

        logger.info(
            "Checkpoint saved at %s",
            checkpoint_dir,
    )

    ####################################################################
    # Resume Training
    ####################################################################

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path,
    ) -> int:

        checkpoint_dir = Path(
            checkpoint_dir
        )

        state_file = (
            checkpoint_dir /
            "training_state.pt"
        )

        if not state_file.exists():

            logger.warning(
                "Checkpoint not found. Starting from scratch."
            )

            self.start_epoch = 1

            return self.start_epoch

        ####################################################
        # Load State
        ####################################################

        state = torch.load(

            state_file,

            map_location=self.device,

        )

        ####################################################
        # Optimizer
        ####################################################

        self.optimizer.load_state_dict(

            state["optimizer_state_dict"]

        )

        ####################################################
        # Scheduler
        ####################################################

        if (

            self.scheduler is not None

            and

            state["scheduler_state_dict"] is not None

        ):

            self.scheduler.load_state_dict(

                state["scheduler_state_dict"]

            )

        ####################################################
        # Restore Metadata
        ####################################################

        self.best_f1 = state["best_f1"]

        self.best_epoch = state["best_epoch"]

        self.start_epoch = (
            state["epoch"] + 1
        )

        logger.info(

            "Training state restored. Resuming from epoch %d.",

            self.start_epoch,

        )

        return self.start_epoch
    ####################################################################
    # History
    ####################################################################

    def save_history(
        self,
    ) -> None:

        json_file = (
            self.checkpoint_dir /
            "training_history.json"
        )

        csv_file = (
            self.checkpoint_dir /
            "training_history.csv"
        )

        with open(
            json_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

        pd.DataFrame(
            self.history
        ).to_csv(
            csv_file,
            index=False,
        )

        logger.info(
            "Training history saved."
        )

    ####################################################################
    # Evaluation
    ####################################################################

    def evaluate(
        self,
        dataloader,
    ) -> Dict:

        validation_loss, metrics = (
            self.validate_epoch(
                dataloader
            )
        )

        print()

        print("=" * 60)

        print("Evaluation")

        print("=" * 60)

        print(
            f"Loss      : {validation_loss:.4f}"
        )

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"Entity F1 : {metrics['f1']:.4f}"
        )

        return metrics        
    
    ####################################################################
    # Prediction
    ####################################################################

    def predict(
        self,
        batch,
    ) -> torch.Tensor:

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(

                input_ids=batch[
                    "input_ids"
                ].to(self.device),

                attention_mask=batch[
                    "attention_mask"
                ].to(self.device),
            )

            predictions = torch.argmax(

                outputs.logits,

                dim=-1,

            )

        return predictions.cpu()
    
    ####################################################################
    # Batch Prediction
    ####################################################################

    def predict_batch(
        self,
        dataloader,
    ) -> List:

        predictions = []

        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(

                dataloader,

                desc="Predict",

            ):

                preds = self.predict(
                    batch
                )

                predictions.extend(
                    preds.numpy()
                )

        return predictions