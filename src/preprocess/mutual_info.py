import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from src.preprocess.base import BasePreprocessor
from src.util.caching import PickleCacheHandler, hash_dataframe


class MutInfoFeatSelection(BaseModel, BasePreprocessor):

    target_column: str
    n_features: Optional[int] = None

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:

        # define Boruta feature selection method

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        if self.n_features is None:
            self.n_features = len(X.columns)

        selector = SelectKBest(mutual_info_classif, k=self.n_features)
        X_sub = selector.fit_transform(X, y)

        df = pd.DataFrame(X_sub, columns=selector.get_feature_names_out(input_features=X.columns))
        df[self.target_column] = y

        return df

    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_hash = hash_dataframe(data=data)

        cache_handler = PickleCacheHandler(
            filepath=Path('mutual_information') / data_hash
        )

        result = cache_handler.read()

        if result is None:
            result = self._select_features(data)
            cache_handler.write(obj=result)

        kwargs['data'] = result

        return kwargs
    
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
        shuffle=False, random_state=42
    )
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y

    result = MutInfoFeatSelection(
        target_column='target',
        n_features=2
    ).execute(data=data)

    print(result['data'])