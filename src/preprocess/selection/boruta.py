import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from typing import Optional
from pathlib import Path

from src.preprocess.base import BaseFeatureSelector
from src.util.caching import PickleCacheHandler, hash_dataframe
from src.util.datasets import DatasetSchema


class BorutaImportance(BaseModel):

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:

        data_hash = hash_dataframe(data=X) + hash_dataframe(data=y)

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / f"{data_hash}"
        )

        result = cache_handler.read()

        if result is None:
            rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

            # define Boruta feature selection method
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

            feat_selector.fit(X.values, y.values)

            result = pd.Series(data=feat_selector.ranking_, index=X.columns)  

            cache_handler.write(obj=result)

        return result


class BorutaFeatSelection(BaseModel, BaseFeatureSelector):

    n_features: Optional[int] = None

    def _select_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.n_features is None:
            return data

        importances = BorutaImportance().get_feature_importance(
            X=data.drop(columns=[DatasetSchema.class_column]),
            y=data[DatasetSchema.class_column]
        )

        importances.sort_values(ascending=False, inplace=True)
        selected_features = importances.head(self.n_features).index.to_list()

        return data[selected_features + [DatasetSchema.class_column]]
