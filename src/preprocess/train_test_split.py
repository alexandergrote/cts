import pandas as pd
from pydantic import BaseModel, model_validator
import numpy as np
from typing import Optional

from src.preprocess.base import BasePreprocessor
from sklearn.model_selection import train_test_split


class TrainTestSplit(BaseModel, BasePreprocessor):
    target_name: str
    split_type: str
    test_size: float
    random_state: int
    split_col: Optional[str]

    @model_validator(mode='after')
    def _validate_split_col(self):

        split_col = self.split_col
        split_type = self.split_type

        if split_type not in ('random_based', 'column_based'):
            raise ValueError("split_type not correctly specified")

        if (split_type == 'column_based') and (split_col is None):
            raise ValueError("split_col must be specified in order to use this split_type")

        return self

    def _split_train_test_random_based(self, data: pd.DataFrame):
        # split data into x and y
        x = data.drop(self.target_name, axis=1)
        y = data[self.target_name]

        # split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        return x_train, x_test, y_train, y_test

    def _split_train_test_column_based(self, data: pd.DataFrame):

        # get unique customer_ids
        customer_id_array = data[self.split_col].unique()

        # Calculate the number of elements to extract depending on the ratio and total number  of distinct ids
        num_elements = len(customer_id_array)
        num_elements_to_extract = int(num_elements * self.test_size)

        # Set the seed for reproducibility (optional)
        np.random.seed(self.random_state)

        # Randomly select the indices for extracting elements
        extracted_elements = np.random.choice(customer_id_array, size=num_elements_to_extract, replace=False)

        mask_test = data[self.split_col].isin(extracted_elements)
        mask_train = mask_test == False

        # split data into train and test sets
        data_train = data[mask_train].reset_index(drop=True)
        data_test = data[mask_test].reset_index(drop=True)

        # split train and test data into x and y
        x_train = data_train.drop(self.target_name, axis=1)
        y_train = data_train[self.target_name]
        x_test = data_test.drop(self.target_name, axis=1)
        y_test = data_test[self.target_name]

        return x_train, x_test, y_train, y_test

    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        mapping = {
            'random_based': self._split_train_test_random_based,
            'column_based': self._split_train_test_column_based
        }

        fun = mapping[self.split_type]

        kwargs['x_train'], kwargs['x_test'], kwargs['y_train'], kwargs['y_test'] = fun(data=data)

        return kwargs
