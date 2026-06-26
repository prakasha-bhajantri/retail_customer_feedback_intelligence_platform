from __future__ import annotations

import re


class FeatureEngineer:

    SENTENCE_PATTERN = re.compile(r"[.!?]+")

    @classmethod
    def generate(cls, review: dict) -> dict:

        text = review["review_text"]

        words = text.split()

        sentence_count = len(
            [s for s in cls.SENTENCE_PATTERN.split(text) if s.strip()]
        )

        review["character_count"] = len(text)

        review["word_count"] = len(words)

        review["sentence_count"] = sentence_count

        review["average_word_length"] = (
            round(
                sum(len(w) for w in words) / len(words),
                2
            )
            if words else 0
        )

        review["contains_question"] = "?" in text

        review["contains_exclamation"] = "!" in text

        review["helpful_ratio"] = (
            review["helpful_yes"] / review["helpful_total"]
            if review["helpful_total"] > 0
            else 0
        )

        return review