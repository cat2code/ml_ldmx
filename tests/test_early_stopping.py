import unittest

from ml_ldmx.train.early_stopping import (
    EarlyStoppingState,
    early_stopping_state_from_history,
)


class EarlyStoppingTest(unittest.TestCase):
    def test_minimum_epoch_floor_and_patience(self):
        state = EarlyStoppingState()
        for completed_epoch, loss in enumerate((1.0, 0.9, 0.91, 0.92, 0.93), start=1):
            state, _improved = state.update(loss, min_delta=1e-4)
            if completed_epoch < 5:
                self.assertFalse(state.should_stop(completed_epoch, min_epochs=5, patience=3))
        self.assertTrue(state.should_stop(completed_epochs=5, min_epochs=5, patience=3))

    def test_significant_improvement_resets_patience(self):
        state = EarlyStoppingState(reference_loss=0.5, bad_epochs=2)
        state, improved = state.update(0.49, min_delta=1e-4)
        self.assertTrue(improved)
        self.assertEqual(state.bad_epochs, 0)
        self.assertEqual(state.reference_loss, 0.49)

    def test_tiny_change_does_not_reset_patience(self):
        state = EarlyStoppingState(reference_loss=0.5, bad_epochs=1)
        state, improved = state.update(0.49995, min_delta=1e-4)
        self.assertFalse(improved)
        self.assertEqual(state.bad_epochs, 2)
        self.assertEqual(state.reference_loss, 0.5)

    def test_resume_state_is_reconstructed_from_history(self):
        history = [
            {"epoch": 1, "val_loss": 1.0},
            {"epoch": 2, "val_loss": 0.8},
            {"epoch": 3, "val_loss": 0.81},
            {"epoch": 4, "val_loss": 0.82},
        ]
        state = early_stopping_state_from_history(history, min_delta=1e-4)
        self.assertEqual(state.reference_loss, 0.8)
        self.assertEqual(state.bad_epochs, 2)


if __name__ == "__main__":
    unittest.main()
