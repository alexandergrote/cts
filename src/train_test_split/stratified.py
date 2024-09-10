import pandas as pd
from pydantic import BaseModel
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit

from src.train_test_split.base import BaseTrainTestSplit
from src.util.datasets import DatasetSchema

class StratifiedSplit(BaseModel, BaseTrainTestSplit):

    test_size: float
    random_state: int

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        data2split = data[[DatasetSchema.id_column, DatasetSchema.class_column]].drop_duplicates()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-self.test_size, random_state=self.random_state)
        generator = sss.split(data2split.values, data2split[DatasetSchema.class_column].values)

        for train_index, test_index in generator:
            data_train = data2split.iloc[train_index][DatasetSchema.id_column]
            data_test = data2split.iloc[test_index][DatasetSchema.id_column]

        # merge with original data
        data_train = data.merge(data_train, left_on=DatasetSchema.id_column, right_on=DatasetSchema.id_column, how='inner')
        data_test = data.merge(data_test, left_on=DatasetSchema.id_column, right_on=DatasetSchema.id_column, how='inner')        

        return data_train, data_test
