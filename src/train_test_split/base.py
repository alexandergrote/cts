import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple

from src.util.custom_logging import Pickler

class BaseTrainTestSplit(ABC):

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError()
    
    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        kwargs['data_train'], kwargs['data_test'] = self.split(data=data)

        return kwargs