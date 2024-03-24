import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from src.util.caching import PickleCacheHandler, hash_dataframe
    
class BaseFeatureEncoder(ABC):

    @abstractmethod
    def _encode(self, *args, **kwargs) -> dict:
        raise NotImplementedError()
    
    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        assert 'case_name' in kwargs

        data_hash = hash_dataframe(data=data)

        case_name = kwargs['case_name'].split('__')[0]

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / f"{case_name}__{data_hash}"
        )

        result = cache_handler.read()

        if result is None:
            result = self._encode(data=data, **kwargs)
            cache_handler.write(obj=result)

        kwargs['data'] = result['data']
        kwargs['rules'] = result.get('rules', None)

        return kwargs
        

class BaseFeatureSelector(ABC):

    @abstractmethod
    def _select_features(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
    
    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_hash = hash_dataframe(data=data)

        if not hasattr(self, 'n_features'):
            raise ValueError("Attribute 'n_features' must be set.")

        feat_id = '__features_'

        if hasattr(self, 'n_features'):

            n_features = getattr(self, 'n_features')

            if n_features is None:
                data_hash += f"{feat_id}all"
            else:
                data_hash += f"{feat_id}{str(getattr(self, 'n_features'))}"

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / data_hash
        )

        result = cache_handler.read()

        if result is None:
            result = self._select_features(data=data, **kwargs)
            cache_handler.write(obj=result)

        kwargs['data'] = result

        return kwargs