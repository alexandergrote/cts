import pandas as pd
from pydantic import BaseModel
from typing import Optional, List

from src.preprocess.base import BaseFeatureEncoder
from src.util.datasets import DatasetSchema


class OneHotEncoder(BaseModel, BaseFeatureEncoder):

    _columns: Optional[List[str]] = None

    def _encode(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        data_copy = data.copy()

        # one hot encode data
        df_pivot = data_copy[[DatasetSchema.id_column, DatasetSchema.event_column]].drop_duplicates().pivot_table(index=DatasetSchema.id_column, columns=DatasetSchema.event_column, aggfunc='size')
        df_pivot = ~df_pivot.isna()
        df_pivot = df_pivot.astype(int)
        df_pivot.index.name = DatasetSchema.id_column
        df_pivot.reset_index(inplace=True)

        # merge maleware back to data
        df_maleware = data_copy[[DatasetSchema.id_column, DatasetSchema.class_column]].drop_duplicates()
        df_pivot = df_pivot.merge(df_maleware, left_on=DatasetSchema.id_column, right_on=DatasetSchema.id_column, how='inner')

        assert df_pivot.shape[0] == data_copy[DatasetSchema.id_column].nunique()
        
        return df_pivot.drop(columns=[DatasetSchema.id_column], errors='ignore')
    
    def _encode_train(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_train_copy = data.copy(deep=True)

        data_train_copy = self._encode(data=data_train_copy)

        self._columns = data_train_copy.columns.to_list()

        return {'data': data_train_copy}

    def _encode_test(self, *, data: pd.DataFrame, **kwargs) -> dict:

        if self._columns is None:
            raise ValueError("Needs to be fitted on training data first prior to encoding test data")
        
        data_test_copy = data.copy(deep=True)

        data_test_copy = self._encode(data=data_test_copy)

        # check for potential values that have not been observed in training data
        columns2drop = [col for col in data_test_copy.columns.to_list() if col not in self._columns]
        data_test_copy.drop(columns=columns2drop)

        return {'data': data_test_copy}


