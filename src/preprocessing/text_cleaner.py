import html
import re


class TextCleaner:

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    EMAIL_PATTERN = re.compile(
        r"\S+@\S+"
    )

    MULTISPACE_PATTERN = re.compile(
        r"\s+"
    )

    @classmethod
    def clean(cls, text: str) -> str:

        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = cls.URL_PATTERN.sub("", text)

        # Remove emails
        text = cls.EMAIL_PATTERN.sub("", text)

        # Collapse whitespace
        text = cls.MULTISPACE_PATTERN.sub(
            " ",
            text
        )

        return text.strip()