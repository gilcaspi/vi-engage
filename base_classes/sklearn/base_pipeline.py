from typing import List, Optional

from base_classes.sklearn.base_step import BaseStep, XType, YType, ReturnType


class BasePipeline(BaseStep[XType, YType, ReturnType]):
    def __init__(self, steps: List[BaseStep]):
        self.steps = steps

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, x: XType, y: Optional[YType] = None) -> ReturnType:
        for step in self.steps[:-1]:
            x = step.run(x, y)
        return self.steps[-1].run(x, y)
