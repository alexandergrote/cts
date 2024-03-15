import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import StratifiedShuffleSplit

from src.train_test_split.base import BaseTrainTestSplit
from src.util.logging import Pickler

class StratifiedSplit(BaseModel, BaseTrainTestSplit):

    target_name: str
    test_size: float
    random_state: int

    def split(self, data: pd.DataFrame):

        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-self.test_size, random_state=self.random_state)
        generator = sss.split(data.values, data[self.target_name].values)

        for train_index, test_index in generator:
            data_train = data.iloc[train_index]
            data_test = data.iloc[test_index]

        x_train, y_train = data_train.drop(self.target_name, axis=1), data_train[self.target_name]
        x_test, y_test = data_test.drop(self.target_name, axis=1), data_test[self.target_name]

        return x_train, x_test, y_train, y_test
