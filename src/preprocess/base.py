import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional

    
class BaseFeatureEncoder(ABC):

    @abstractmethod
    def _encode_train(self, *, data: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()
    
    @abstractmethod
    def _encode_test(self, *, data: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()
    
    def execute(self, *, data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs) -> dict:

        result_train: dict = self._encode_train(data=data_train, **kwargs)
        data_train_processed = result_train['data']

        assert len(data_train_processed) > 0

        rules = result_train.get('rules', {})
        kwargs['rules'] = rules
        
        result_test: dict = self._encode_test(data=data_test, **kwargs)
        data_test_processed = result_test['data']


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

        data_train_processed = self._select_features_train(data=data_train, **kwargs)
        data_test_processed = self._select_features_test(data=data_test, **kwargs)

        kwargs['data_train'] = data_train_processed
        kwargs['data_test'] = data_test_processed

        return kwargs
