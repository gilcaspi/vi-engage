import torch
from torch import nn

class BasePreprocessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the forward() method."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"
