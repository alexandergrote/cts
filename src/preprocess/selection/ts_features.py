import pandas as pd
from pydantic import BaseModel
from typing import Optional

from src.preprocess.base import BaseFeatureSelector
from src.preprocess.util.datasets import DatasetUniqueRules, DatasetUniqueRulesSchema
from src.preprocess.util.rules import RuleEncoder


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
        
        ranked_rules = rules.rank_rules(
            criterion=DatasetUniqueRulesSchema.delta_confidence,
            weighted_by_support=True
        )

        # get top n_features rules and format them correctly to use them with the dataframe
        top_rules = [RuleEncoder.encode_rule_id(rule=el[0]) for el in ranked_rules[:self.n_features]]
        
        self._columns = top_rules + [self.target_column]

        return data_copy[self._columns]
