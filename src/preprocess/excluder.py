import pandas as pd
from pydantic import BaseModel
from typing import Optional, List

from src.preprocess.base import BasePreprocessor


class ColumnExcluder(BaseModel, BasePreprocessor):

    # functions to exclude columns of specific types (such as date & time)
    columns: Optional[List[str]] = None

    exclude_all_datetime: bool = False
    exclude_all_ids: bool = False
    exclude_all_date_expressions: bool = False

    exclude_all_expressions: Optional[List[str]] = None

    exceptions: Optional[List[str]] = None

    def execute(self, x_train: pd.DataFrame, x_test: pd.DataFrame, **kwargs) -> dict:

        x_train_copy = x_train.copy()
        x_test_copy = x_test.copy()

        # placeholder for columns to exclude
        columns = []

        # override columns if specified
        if self.columns is not None:
            columns = self.columns

        if self.exclude_all_datetime:
            columns += x_train_copy.select_dtypes(include=['datetime']).columns.tolist()

        if self.exclude_all_ids:
            columns += x_train_copy.filter(like='_id').columns.tolist()

        if self.exclude_all_date_expressions:
            columns += x_train_copy.filter(like='_date').columns.tolist()

        if self.exclude_all_expressions is not None:
            for expression in self.exclude_all_expressions:
                columns += x_train_copy.filter(like=expression).columns.to_list()

        # remove exceptions from list if they are in the columns list
        if self.exceptions is not None:
            columns = [col for col in columns if col not in self.exceptions]

        # drop columns
        x_train_copy = x_train_copy.drop(columns=columns)
        x_test_copy = x_test_copy.drop(columns=columns)

        # update kwargs
        kwargs['x_train'] = x_train_copy
        kwargs['x_test'] = x_test_copy

        return kwargs
