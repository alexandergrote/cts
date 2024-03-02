import pandas as pd
from abc import ABC, abstractmethod


class BaseProcessModel(ABC):

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, x_test: pd.DataFrame, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        raise NotImplementedError()
