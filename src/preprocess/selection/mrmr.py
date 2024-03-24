# inspired by https://github.com/smazzanti/mrmr/blob/main/mrmr/pandas.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, field_validator
from typing import Optional, List, Callable

from src.preprocess.base import BaseFeatureSelector
from src.preprocess.selection.random_forest import RFImportance
from src.preprocess.util.correlation import theils_u
from src.util.caching import PickleCacheHandler, hash_dataframe, hash_string

FLOOR = .001

def random_forest_classif(X, y):
    return RFImportance().get_feature_importance(X, y)

def theils_u_wrapper(target_column: str, features: List[str], X: pd.DataFrame) -> pd.Series:
    """
    Calculate the Theil's U value for a target column and a list of features.

    Parameters:
    - target_column (str): the target column
    - features (List[str]): list of features
    - X (pd.DataFrame): the dataframe

    Returns:
    - pd.Series: a series with the Theil's U value and the features as indices
    """

    data_hash = hash_dataframe(data=X) + hash_string(target_column) + hash_string('__'.join(features))

    cache_handler = PickleCacheHandler(
        filepath=Path('theils_u_wrapper') / f"{data_hash}"
    )

    result = cache_handler.read()

    if result is None:
            
            scores = [theils_u(x=X[feature], y=X[target_column]) for feature in features]
            result = pd.Series(data=scores, index=features)
            cache_handler.write(obj=result)

    return result

class MRMRFeatSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    perc_features: Optional[float] = None

    relevance_func: Callable = random_forest_classif
    redundancy_func: Callable = theils_u_wrapper
    denominator_func: Callable = np.mean

    @field_validator('perc_features')
    def check_perc_features(cls, v):

        if v is None:
            return v

        if v > 1.0:
            return v / 100
        return v
    
    def _mrmr(self, X: pd.DataFrame, y: pd.Series, n: int) -> List[str]:

        relevance = self.relevance_func(X, y)
        features = relevance[relevance.fillna(0) > 0].index.to_list()
        relevance = relevance.loc[features]
        redundancy = pd.DataFrame(FLOOR, index=features, columns=features)

        K = min(n, len(features))

        selected_features: List[str] = []
        not_selected_features = features.copy()

        for i in tqdm(range(K)):

            score_numerator = relevance.loc[not_selected_features]

            if i > 0:

                last_selected_feature = selected_features[-1]
                not_selected_features_sub = not_selected_features

                if not_selected_features_sub:
                    redundancy.loc[not_selected_features_sub, last_selected_feature] = self.redundancy_func(
                        target_column=last_selected_feature,
                        features=not_selected_features_sub,
                        X=X
                    ).fillna(FLOOR).abs().clip(FLOOR)
                    score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                        self.denominator_func, axis=1).replace(1.0, float('Inf'))

            else:
                score_denominator = pd.Series(1, index=features)

            score = score_numerator / score_denominator

            best_feature = score.index[score.argmax()]
            selected_features.append(best_feature)
            not_selected_features.remove(best_feature)

        
        return selected_features
         

    def _select_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.perc_features is None:
            return data
        
        # find all relevant features - 5 features should be selected
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        n_features = int(self.perc_features * len(X.columns))
        
        selected_features = self._mrmr(X, y, n_features)

        return data[selected_features + [self.target_column]]