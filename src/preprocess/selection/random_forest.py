import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.util.caching import PickleCacheHandler, hash_dataframe
from src.preprocess.base import BaseFeatureSelector

class RFImportance(BaseModel):

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:

        data_hash = hash_dataframe(data=X) + hash_dataframe(data=y)

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / f"{data_hash}"
        )

        result = cache_handler.read()

        if result is None:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            result = pd.Series(rf.feature_importances_, index=X.columns)
            cache_handler.write(obj=result)

        return result

class RFFeatSelection(BaseModel, BaseFeatureSelector):

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

        # train random forest
        rf_importance = RFImportance()
        feature_importance = rf_importance.get_feature_importance(X, y)
        feature_importance.sort_values(ascending=False, inplace=True)

        n_features = int(self.perc_features * len(feature_importance))
        selected_features = feature_importance.head(n_features).index.tolist()

        return data[selected_features + [self.target_column]]

    
    
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
        shuffle=False, random_state=42
    )
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y

    result = RFFeatSelection(
        target_column='target',
        n_features=200
    ).execute(data=data)

    print(result['data'])