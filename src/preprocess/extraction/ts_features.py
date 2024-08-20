from .v1.ts_features import *

import pandas as pd
from pydantic import  BaseModel
from pydantic.config import ConfigDict
from typing import List, Iterable


class Sequence(BaseModel):
    id_values: List[str]
    sequence_values: List[str]


class PrefixSpanDataset(BaseModel):

    id_columns: List[str]
    event_column: str
    raw_data: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_sequences(self) -> List[Sequence]:

        data_copy = self.raw_data.copy(deep=True)

        # ensure that the events are strings
        data_copy[self.event_column] = data_copy[self.event_column].astype(str)

        # since we are interested in calculating confidence of rules
        # we need at least a sequence with length 2
        # as a first step, we can remove all sequences with length 1
        sequence_nunique = data_copy.groupby(self.id_columns)[self.event_column].nunique() != 1
        sequence_nunique.name = 'to_keep'
        data_copy = data_copy.merge(sequence_nunique, left_on=self.id_columns, right_index=True)
        data_copy = data_copy[data_copy['to_keep'] == True]
        sequences = data_copy.groupby(self.id_columns)[self.event_column].apply(list)

        return [Sequence(id_values=list(index), sequence_values=value) for index, value in sequences.items()]