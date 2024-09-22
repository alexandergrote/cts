import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.util.caching import PickleCacheHandler, hash_dataframe
    
class BaseFeatureEncoder(ABC):

    @abstractmethod
    def _encode_train(self, *, data: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()
    
    @abstractmethod
    def _encode_test(self, *, data: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()
    
    def execute(self, *, data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs) -> dict:

        data_hash_train = hash_dataframe(data=data_train)
        data_hash_test = hash_dataframe(data=data_test)
        data_hash = data_hash_train + data_hash_test

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / f"{data_hash}"
        )

        result = cache_handler.read()

        if True:

            result_train: dict = self._encode_train(data=data_train, **kwargs)
            data_train_processed = result_train['data']

            assert len(data_train_processed) > 0

            rules = result_train.get('rules', {})
            kwargs['rules'] = rules
            
            result_test: dict = self._encode_test(data=data_test, **kwargs)
            data_test_processed = result_test['data']
            cache_handler.write(obj=(data_train_processed, data_test_processed, kwargs['rules']))
        else:
            data_train_processed, data_test_processed, rules = result


        kwargs['data_train'] = data_train_processed
        kwargs['data_test'] = data_test_processed
        kwargs['rules'] = rules

        return kwargs
        

class BaseFeatureSelector(ABC):

    _columns: Optional[List[str]] = None

    @abstractmethod
    def _select_features_train(self, data: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()

    def _select_features_test(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self._columns is None:
            raise ValueError("Needs to be fitted on training data first prior to encoding test data")

        columns2add = [col for col in self._columns if col not in data.columns]
        
        for col in columns2add:
            data[col] = 0

        return data[self._columns]
    
    def execute(self, *, data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs) -> dict:

        data_hash_train = hash_dataframe(data=data_train)
        data_hash_test = hash_dataframe(data=data_test)
        data_hash = data_hash_train + data_hash_test

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

        if True:
            data_train_processed = self._select_features_train(data=data_train, **kwargs)
            data_test_processed = self._select_features_test(data=data_test, **kwargs)
            
            cache_handler.write(obj=(data_train_processed, data_test_processed))
        else:
            data_train_processed, data_test_processed = result

        kwargs['data_train'] = data_train_processed
        kwargs['data_test'] = data_test_processed

        return kwargs
