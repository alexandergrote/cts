import pandas as pd
import numpy as np
from pydantic import BaseModel

from src.preprocess.base import BaseFeatureEncoder


class LSTMSequence(BaseModel, BaseFeatureEncoder):

    id_column: str
    feature_column: str
    time_column: str
    target_column: str  

    def _encode(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_copy = data.copy()

        # one hot encode data
        mapping = {event: i+1 for i, event in enumerate(data_copy[self.feature_column].unique())}
        data_copy[self.feature_column] = data_copy[self.feature_column].map(mapping)

        data_copy.sort_values(by=self.time_column, inplace=True)

        sequences = data_copy.groupby(self.id_column)[self.feature_column].apply(list).to_list()
        targets = data_copy.groupby(self.id_column)[self.target_column].apply(lambda x: np.unique(x)[0]).to_list()

        # summarize in one dataframe
        result = pd.DataFrame(sequences)
        result[self.target_column] = targets

        kwargs['data'] = result
        
        return kwargs
