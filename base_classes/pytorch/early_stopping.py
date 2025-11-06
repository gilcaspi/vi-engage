from enum import StrEnum
from typing import Optional


class EarlyStoppingMode(StrEnum):
    MIN = 'min'
    MAX = 'max'


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: EarlyStoppingMode = EarlyStoppingMode.MIN):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return

        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:  # mode == 'max'
            return current_score > self.best_score + self.min_delta

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False