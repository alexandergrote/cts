import sklearn.linear_model
import xgboost
import yaml
from pydantic import BaseModel, model_validator
from typing import Optional, Any
from pathlib import Path
import pandas as pd

from src.model.base import BaseProcessModel
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory
from src.util.custom_logging import Pickler


class SklearnModel(BaseModel, BaseProcessModel):

    model: Optional[Any]
    params: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'

    @model_validator(mode='after')
    def _init_model(self):

        # get model name
        model_name = self.model

        # init model
        model = DynamicImport.init_class(model_name, params=self.params)

        # overwrite model
        self.model = model

        return self

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, **kwargs):

        self.model.fit(x_train, y_train)

        feature_column = 'feature_names'
        importance_column = 'importance'

        importance_df = None

        if isinstance(self.model, xgboost.XGBClassifier):

            feature_importance = self.model.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame(feature_importance, index=[0]).T.reset_index()
            importance_df.columns = [feature_column, importance_column]
            print(importance_df.sort_values(by=importance_column, ascending=False))

        if isinstance(self.model, sklearn.linear_model.LogisticRegression):

            importance_df = pd.DataFrame({
                feature_column: self.model.feature_names_in_,
                importance_column: self.model.coef_.ravel()
            })

        # save results
        Pickler.write(importance_df, output_dir / 'importance.pickle')
        Pickler.write(self.model, output_dir / 'model.pickle')

        kwargs['x_train'] = x_train
        kwargs['y_train'] = y_train

        return kwargs

    # predict(self, data: pd.DataFrame):
    def _predict(self, x_test, **kwargs):
        return self.model.predict(x_test)


    # predict(self, data: pd.DataFrame):
    def _predict_proba(self, x_test, **kwargs):
        return self.model.predict_proba(x_test)


if __name__ == '__main__':

    # read tft config
    with open(Directory.CONFIG / "model/nb.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # data preprocessing
    # demo classification
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True, as_frame=True)
    y.name = 'species'

    # demo classification
    model = SklearnModel(**config['params'])

    model.fit(x_train=X, y_train=y)

    pred = model.predict(X)
    print(pred)

    pred_proba = model.predict_proba(X)
    print(pred_proba)
