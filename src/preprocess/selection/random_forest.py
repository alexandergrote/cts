import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Optional, List
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.util.caching import PickleCacheHandler, hash_dataframe
from src.preprocess.base import BaseFeatureSelector
from src.util.datasets import DatasetSchema

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

    n_features: Optional[int] = None

    def _select_features_train(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.n_features is None:

            self._columns = data.columns.to_list()

            return data

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[DatasetSchema.class_column])
        y = data[DatasetSchema.class_column]

        # train random forest
        rf_importance = RFImportance()
        feature_importance = rf_importance.get_feature_importance(X, y)
        feature_importance.sort_values(ascending=False, inplace=True)

        selected_features = feature_importance.head(self.n_features).index.tolist()

        self._columns = selected_features + [DatasetSchema.class_column]

        return data[self._columns]
    
    
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
        shuffle=False, random_state=42
    )
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data[DatasetSchema.class_column] = y

    result = RFFeatSelection(
        n_features=200
    ).execute(data=data)

    print(result)