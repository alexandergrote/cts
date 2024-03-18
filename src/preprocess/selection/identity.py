import pandas as pd
from pydantic import BaseModel

from src.preprocess.base import BaseFeatureSelector


class FeatureIdentity(BaseModel, BaseFeatureSelector):

    def _select_features(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data
        