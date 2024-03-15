import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from src.util.caching import PickleCacheHandler, hash_dataframe

class BasePreprocessor(ABC):

    @abstractmethod
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError()
    

class BaseFeatureSelector(ABC):

    @abstractmethod
    def _select_features(self, *args, **kwargs) -> dict:
        raise NotImplementedError()
    
    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_hash = hash_dataframe(data=data)

        feat_id = '__features_'
        if getattr(self, 'n_features') is not None:
            data_hash += f"{feat_id}{str(self.n_features)}"
        else:
            data_hash += f'{feat_id}all' 

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / data_hash
        )

        result = cache_handler.read()

        if result is None:
            result = self._select_features(data)
            cache_handler.write(obj=result)

        kwargs['data'] = result

        return kwargs