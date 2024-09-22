import yaml
from pydantic import BaseModel, model_validator
from typing import Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

from src.model.base import BaseProcessModel
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory
from src.util.custom_logging import Pickler


class RandomModel(BaseModel, BaseProcessModel):

    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'


    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, **kwargs):

        kwargs['x_train'] = x_train
        kwargs['y_train'] = y_train

        return kwargs

    # predict(self, data: pd.DataFrame):
    def _predict(self, x_test, **kwargs):

        if not isinstance(x_test, pd.DataFrame):
            raise ValueError('x_test must be a pandas dataframe')
        
        # generate random binary predictions
        return np.random.randint(0, 2, size=x_test.shape[0])



    def _predict_proba(self, x_test, **kwargs):
        
        if not isinstance(x_test, pd.DataFrame):
            raise ValueError('x_test must be a pandas dataframe')
        
        # Generate random probability values for two classes
        proba = np.random.rand(x_test.shape[0], 2)
        
        # Ensure the probabilities sum up to 1
        proba[:, 0] = 1 - proba[:, 1]
        
        return proba


        
        

if __name__ == '__main__':

    from tempfile import TemporaryDirectory

    output_dir = TemporaryDirectory()

    # read tft config
    with open(Directory.CONFIG / "model/random_chance.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # data preprocessing
    # demo classification
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True, as_frame=True)
    y.name = 'species'

    params = config['params'] if config['params'] is not None else {}

    # demo classification
    model = RandomModel(**params)

    model.fit(x_train=X, y_train=y, output_dir=output_dir)

    pred = model.predict(X)
    print(pred)

    pred_proba = model.predict_proba(X)
    print(pred_proba)
