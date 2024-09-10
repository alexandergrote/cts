import pandas as pd
import numpy as np
from pydantic import BaseModel

from src.preprocess.base import BaseFeatureEncoder
from src.util.datasets import DatasetSchema


class LSTMSequence(BaseModel, BaseFeatureEncoder):

    mapping: dict = {}

    def _encode_train(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_copy = data.copy()

        # one hot encode data
        self.mapping = {event: i+1 for i, event in enumerate(data_copy[DatasetSchema.event_column].unique())}
        data_copy[DatasetSchema.event_column] = data_copy[DatasetSchema.event_column].map(self.mapping)

        data_copy.sort_values(by=DatasetSchema.time_column, inplace=True)

        sequences = data_copy.groupby(DatasetSchema.id_column)[DatasetSchema.event_column].apply(list).to_list()
        targets = data_copy.groupby(DatasetSchema.id_column)[DatasetSchema.class_column].apply(lambda x: np.unique(x)[0]).to_list()

        # summarize in one dataframe
        result = pd.DataFrame(sequences)
        result[DatasetSchema.class_column] = targets

        kwargs['data'] = result
        
        return kwargs
    
    def _encode_test(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_copy = data.copy()

        data_copy[DatasetSchema.event_column] = data_copy[DatasetSchema.event_column].map(self.mapping)

        # necessary because some values might be missing and therefore be encoding as NaN, which is a float
        data_copy[DatasetSchema.event_column] = data_copy[DatasetSchema.event_column].fillna(0).astype(int)  
        
        data_copy.sort_values(by=DatasetSchema.time_column, inplace=True)

        sequences = data_copy.groupby(DatasetSchema.id_column)[DatasetSchema.event_column].apply(list).to_list()
        targets = data_copy.groupby(DatasetSchema.id_column)[DatasetSchema.class_column].apply(lambda x: np.unique(x)[0]).to_list()

        # summarize in one dataframe
        result = pd.DataFrame(sequences)
        result[DatasetSchema.class_column] = targets

        return {'data': result}
        
