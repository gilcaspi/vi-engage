from typing import Union

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_proba: bool = False) -> Union[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(x)

        if return_proba:
            return torch.softmax(logits, dim=1)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_on_loader(self, loader, device):
        self.eval()
        all_predictions = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            outputs = self.forward(x)

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu().numpy())
        return np.concatenate(all_predictions)
