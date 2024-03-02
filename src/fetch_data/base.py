from abc import ABC, abstractmethod

import pandas as pd


class BaseDataLoader(ABC):

    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError()


class BaseDataset(ABC):

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        raise NotImplementedError()