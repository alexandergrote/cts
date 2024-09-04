import pandas as pd
from pydantic import BaseModel
from typing import Optional

from src.preprocess.base import BaseFeatureSelector
from src.preprocess.util.datasets import DatasetUniqueRules, DatasetUniqueRulesSchema


class TimeSeriesFeatureSelection(BaseModel, BaseFeatureSelector):

    target_column: str
    n_features: Optional[int] = None
    splitting_symbol: str

    def _select_features_train(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        assert 'rules' in kwargs, "Rules must be provided to the feature selector"
        rules = kwargs['rules']
        assert isinstance(rules, DatasetUniqueRules), "Rules must be a DatasetUniqueRules"
        
        data_copy = data.copy(deep=True)

        if self.n_features is None:
            self._columns = data.columns.to_list()
            return data
        
        all_rules = data_copy.filter(like=self.splitting_symbol).columns.to_list()
        rules = rules[rules['index'].isin(all_rules)]
        rules.sort_values(by='mean_ranking', ascending=False, inplace=True)

        important_sequences = rules['index'].head(self.n_features).to_list()
        
        self._columns = important_sequences + [self.target_column]

        return data_copy[self._columns]
