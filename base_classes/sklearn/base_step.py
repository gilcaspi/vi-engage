from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

XType = TypeVar("XType")
YType = TypeVar("YType")
ReturnType = TypeVar("ReturnType")

class BaseStep(ABC, Generic[XType, YType, ReturnType]):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this step."""
        return ''

    @abstractmethod
    def run(self, x: XType, y: Optional[YType] = None) -> ReturnType:
        """Execute the step logic."""
        return ReturnType