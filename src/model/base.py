import pandas as pd
from abc import ABC, abstractmethod


class BaseProcessModel(ABC):

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def _predict(self, x_test: pd.DataFrame, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def _predict_proba(self, x_test: pd.DataFrame, **kwargs):
        raise NotImplementedError()
    
    # predict(self, data: pd.DataFrame):
    def predict(self, x_test, **kwargs):

        pred = self._predict(x_test)

        kwargs['y_pred'] = pred
        kwargs['x_test'] = x_test

        return kwargs

    # predict(self, data: pd.DataFrame):
    def predict_proba(self, x_test, **kwargs):
        pred_proba = self._predict_proba(x_test)
        kwargs['y_pred_proba'] = pred_proba
        kwargs['x_test'] = x_test

        return kwargs
