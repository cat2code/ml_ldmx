"""Small, resume-safe early-stopping helpers."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EarlyStoppingState:
    """Track significant validation-loss improvements across completed epochs."""

    reference_loss: float = math.inf
    bad_epochs: int = 0

    def update(self, validation_loss: float, min_delta: float):
        validation_loss = float(validation_loss)
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        improved = math.isfinite(validation_loss) and (
            validation_loss < self.reference_loss - min_delta
        )
        if improved:
            return EarlyStoppingState(validation_loss, 0), True
        return EarlyStoppingState(self.reference_loss, self.bad_epochs + 1), False

    def should_stop(self, completed_epochs: int, min_epochs: int, patience: int) -> bool:
        """Return whether training should stop after ``completed_epochs``."""
        if min_epochs <= 0:
            raise ValueError("min_epochs must be positive.")
        if patience < 0:
            raise ValueError("patience must be non-negative.")
        return patience > 0 and completed_epochs >= min_epochs and self.bad_epochs >= patience


def early_stopping_state_from_history(history, min_delta: float) -> EarlyStoppingState:
    """Reconstruct early-stopping state so checkpoint resumes remain deterministic."""
    state = EarlyStoppingState()
    for epoch_metrics in history:
        if "val_loss" not in epoch_metrics:
            continue
        state, _improved = state.update(epoch_metrics["val_loss"], min_delta)
    return state
