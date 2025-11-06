from typing import Any, Callable, Optional

import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("Subclasses must implement __getitem__")
