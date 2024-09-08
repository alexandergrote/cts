import pandas as pd

from abc import ABC, abstractmethod

from src.util.datasets import Dataset, DatasetSchema

class BaseDataLoader(ABC):

    def execute(self) -> dict:

        df = self.get_data()

        DatasetSchema.validate(df)
        
        return {'data': df}
    
    @abstractmethod
    def get_data(self) -> Dataset:
        raise NotImplementedError()
    