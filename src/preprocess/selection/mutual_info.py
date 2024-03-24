import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Optional
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from src.preprocess.base import BaseFeatureSelector


class MutInfoFeatSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    perc_features: Optional[float] = None

    @field_validator('perc_features')
    def check_perc_features(cls, v):

        if v is None:
            return v

        if v > 1.0:
            return v / 100
        return v

    def _select_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.perc_features is None:
            return data

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        n_features = int(self.perc_features * len(X.columns))
        selector = SelectKBest(mutual_info_classif, k=n_features)
        X_sub = selector.fit_transform(X, y)
        df = pd.DataFrame(X_sub, columns=selector.get_feature_names_out(input_features=X.columns))
        df[self.target_column] = y

        return df

    
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
        n_features=200
    ).execute(data=data)

    print(result['data'])