import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Extra

from src.preprocess.base import BasePreprocessor


class LabelingFromStatic(BaseModel, BasePreprocessor):

    # static columns
    id_column_static: str
    target_column_static: str

    # event columns
    id_column_event: str

    def _verify_incoming_data(self, case: pd.DataFrame, event: pd.DataFrame):

        # check for column existence
        assert self.id_column_static in case.columns
        assert self.target_column_static in case.columns
        assert self.id_column_event in event.columns

        # check if target_column_static already exists in case columns
        assert self.target_column_static not in event.columns

        # check for dataframe lengths
        for df in [case, event]:
            assert len(df) > 0

    def execute(self, *, event: pd.DataFrame, case: Optional[pd.DataFrame] = None, **kwargs) -> dict:

        # data qa checks
        self._verify_incoming_data(case, event)

        # work on copies
        case_copy = case.copy(deep=True)
        event_copy = event.copy(deep=True)

        # get columns of interest from case df
        case_copy = case_copy[[self.id_column_static, self.target_column_static]]

        # merge with time series data
        event_copy = event_copy.merge(case_copy, left_on=self.id_column_event, right_on=self.id_column_static)

        kwargs['case'] = case
        kwargs['event'] = event_copy

        return kwargs
