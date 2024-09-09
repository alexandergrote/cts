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


class BaseResultDataloader(ABC):

    @abstractmethod
    def _load(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def execute(self, **kwargs) -> dict:

        assert "experiment_name" in kwargs, "experiment_name must be provided."

        kwargs['result_data'] = self._load(**kwargs)

        print("-"*25)
        print("Experiment:", kwargs["experiment_name"])
        print(kwargs['result_data'])

        return kwargs