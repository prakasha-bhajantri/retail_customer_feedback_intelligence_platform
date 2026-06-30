"""
predictor.py

Inference for Retail Named Entity Recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from src.nlp.retail_ner.labels import (
    ID2LABEL,
)


@dataclass
class Entity:

    text: str

    label: str

    confidence: float


class RetailNERPredictor:

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):

        self.device = device or (

            "mps"

            if torch.backends.mps.is_available()

            else

            "cuda"

            if torch.cuda.is_available()

            else

            "cpu"

        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = (

            AutoModelForTokenClassification
            .from_pretrained(
                model_path
            )
            .to(self.device)
        )

        self.model.eval()

    ########################################################

    @torch.no_grad()
    def predict(
        self,
        text: str,
    ):

        encoding = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=256,

        )

        encoding = {

            k: v.to(self.device)

            for k, v in encoding.items()

        }

        outputs = self.model(

            **encoding

        )

        probabilities = (

            outputs.logits.softmax(

                dim=-1

            )

        )

        predictions = (

            probabilities.argmax(

                dim=-1

            )[0]

        )

        confidence = (

            probabilities.max(

                dim=-1

            ).values[0]

        )

        tokens = self.tokenizer.convert_ids_to_tokens(

            encoding["input_ids"][0]

        )

        entities = []

        current_tokens = []

        current_label = None

        current_scores = []

        for token, label_id, score in zip(

            tokens,

            predictions,

            confidence,

        ):

            label = ID2LABEL[

                label_id.item()

            ]

            if token in {

                "[CLS]",

                "[SEP]",

                "[PAD]",

            }:

                continue

            token = token.replace(

                "##",

                "",

            )

            if label == "O":

                if current_tokens:

                    entities.append(

                        Entity(

                            text=" ".join(

                                current_tokens

                            ),

                            label=current_label,

                            confidence=round(

                                sum(

                                    current_scores

                                )

                                /

                                len(

                                    current_scores

                                ),

                                4,

                            ),

                        )

                    )

                    current_tokens = []

                    current_scores = []

                    current_label = None

                continue

            prefix, entity = label.split(

                "-",

                1,

            )

            if prefix == "B":

                if current_tokens:

                    entities.append(

                        Entity(

                            text=" ".join(

                                current_tokens

                            ),

                            label=current_label,

                            confidence=round(

                                sum(

                                    current_scores

                                )

                                /

                                len(

                                    current_scores

                                ),

                                4,

                            ),

                        )

                    )

                current_tokens = [

                    token

                ]

                current_scores = [

                    score.item()

                ]

                current_label = entity

            else:

                current_tokens.append(

                    token

                )

                current_scores.append(

                    score.item()

                )

        if current_tokens:

            entities.append(

                Entity(

                    text=" ".join(

                        current_tokens

                    ),

                    label=current_label,

                    confidence=round(

                        sum(

                            current_scores

                        )

                        /

                        len(

                            current_scores

                        ),

                        4,

                    ),

                )

            )

        return entities