import numpy as np
from abc import ABC, abstractmethod
from typing import List


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs):
        raise NotImplementedError()
