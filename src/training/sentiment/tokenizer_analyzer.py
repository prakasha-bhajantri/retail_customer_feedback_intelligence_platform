from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


class TokenizerAnalyzer:

    def __init__(
        self,
        parquet_path: str,
        model_name: str = "bert-base-uncased",
    ):

        self.df = pd.read_parquet(parquet_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze(
        self,
        sample_size: int = 50000,
    ):

        print("Sampling reviews...")

        sample = self.df.sample(
            min(sample_size, len(self.df)),
            random_state=42,
        )

        lengths = []

        for text in sample["review_text"]:

            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )

            lengths.append(len(tokens))

        lengths = np.array(lengths)

        report = {

            "mean": round(lengths.mean(), 2),

            "median": int(np.median(lengths)),

            "95_percentile": int(
                np.percentile(lengths, 95)
            ),

            "99_percentile": int(
                np.percentile(lengths, 99)
            ),

            "maximum": int(lengths.max()),
        }

        return report