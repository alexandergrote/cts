import pandas as pd
from pydantic import BaseModel
from typing import Optional

from src.preprocess.base import BaseFeatureSelector


class FeatureIdentity(BaseModel, BaseFeatureSelector):

    n_features: Optional[int] = None

    def _select_features(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data
        