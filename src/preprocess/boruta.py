import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

from src.preprocess.base import BaseFeatureSelector


class BorutaFeatSelection(BaseModel, BaseFeatureSelector):

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