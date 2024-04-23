import pandas as pd
from pydantic import BaseModel
from typing import Optional

from src.preprocess.base import BaseFeatureSelector


class FeatureIdentity(BaseModel, BaseFeatureSelector):

    n_features: Optional[int] = None

    def _select_features_train(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        self._columns = data.columns.to_list()
        return data