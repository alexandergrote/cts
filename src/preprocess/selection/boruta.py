import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from typing import Optional

from src.preprocess.base import BaseFeatureSelector


class BorutaFeatSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    perc_features: Optional[float] = None

    @field_validator('perc_features')
    def check_perc_features(cls, v):

        if v is None:
            return v

        if v > 1.0:
            print('-'*100)
            print('v:', v)
            print('-'*100)

            return v / 100
        return v

    def _select_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.perc_features is None:
            return data

        rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        
        y = data[self.target_column]

        feat_selector.fit(X.values, y.values)

        df_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': feat_selector._get_imp(X, y)
        })

        n_features = int(self.perc_features * len(df_importances))

        df_importances = df_importances.sort_values(by='importance', ascending=False)
        selected_features = df_importances['feature'].head(n_features).to_list()

        return data[selected_features + [self.target_column]]
