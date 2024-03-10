import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

from src.preprocess.base import BasePreprocessor
from src.util.caching import PickleCacheHandler, hash_dataframe


class BorutaFeatSelection(BaseModel, BasePreprocessor):

    target_column: str

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:

        rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        
        y = data[self.target_column]

        feat_selector.fit(X.values, y.values)

        # call transform() on X to filter it down to selected features
        X_sub = feat_selector.transform(X.values)

        df = pd.DataFrame(X_sub, columns=X.columns[feat_selector.support_])
        df[self.target_column] = y

        return df

    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_hash = hash_dataframe(data=data)

        cache_handler = PickleCacheHandler(
            filepath=Path('boruta') / data_hash
        )

        result = cache_handler.read()

        if result is None:
            result = self._select_features(data)
            cache_handler.write(obj=result)

        kwargs['data'] = result

        return kwargs