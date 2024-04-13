import pandas as pd
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from sklearn.feature_selection import mutual_info_classif

from src.util.caching import hash_dataframe, PickleCacheHandler
from src.preprocess.base import BaseFeatureSelector

class MutInfoImportance(BaseModel):

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:

        data_hash = hash_dataframe(data=X) + hash_dataframe(data=y)

        cache_handler = PickleCacheHandler(
            filepath=Path(self.__class__.__name__) / f"{data_hash}"
        )

        result = cache_handler.read()

        if result is None:
                
            result = pd.Series(mutual_info_classif(X, y), index=X.columns)
            cache_handler.write(obj=result)

        return result



class MutInfoFeatSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    n_features: Optional[int] = None

    def _select_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.n_features is None:
            return data

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]    

        mut_info = MutInfoImportance()
        feature_importance = mut_info.get_feature_importance(X, y)
        feature_importance.sort_values(ascending=False, inplace=True)
        
        selected_features = feature_importance.head(self.n_features).index.tolist()

        return data[selected_features + [self.target_column]]

    
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=2,
        shuffle=False, random_state=42
    )
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y

    result = MutInfoFeatSelection(
        target_column='target',
        n_features=200
    ).execute(data=data)

    print(result['data'])