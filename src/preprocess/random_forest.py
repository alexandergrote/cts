import pandas as pd
from pydantic import BaseModel
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from src.preprocess.base import BaseFeatureSelector


class RFFeatSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    n_features: Optional[int] = None

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:

        # define Boruta feature selection method

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        if self.n_features is None:
            self.n_features = len(X.columns)

        # train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        df_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
        df_importances = df_importances.reset_index().sort_values(by='importance', ascending=False)

        selected_features = df_importances['index'].head(self.n_features).tolist()

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