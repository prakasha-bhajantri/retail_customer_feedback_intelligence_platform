import logging
import sys

logger = logging.getLogger("retail-feedback")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console)