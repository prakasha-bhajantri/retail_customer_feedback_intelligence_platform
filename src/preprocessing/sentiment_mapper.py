"""
sentiment_mapper.py

Maps Amazon star ratings to sentiment labels.
"""

class SentimentMapper:
    """
    Maps ratings to sentiment classes.

    1-2 -> negative
    3   -> neutral
    4-5 -> positive
    """

    @staticmethod
    def map(rating: float) -> str:
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"