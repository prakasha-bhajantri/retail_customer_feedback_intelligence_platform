"""
scheduler.py

Learning rate schedulers.
"""

from transformers import get_linear_schedule_with_warmup


class SchedulerFactory:

    @staticmethod
    def linear_warmup(
        optimizer,
        total_training_steps: int,
        warmup_ratio: float,
    ):

        warmup_steps = int(
            total_training_steps * warmup_ratio
        )

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )