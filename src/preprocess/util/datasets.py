import pandas as pd
import pandera as pa

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from pandera.typing import DataFrame, Series

from src.preprocess.util.types import AnnotatedSequence, FrequentPatternWithConfidence


class DatasetSchema(pa.DataFrameModel):
    event_column: Series[str] = pa.Field(coerce=True)
    time_column: Series[int]
    class_column: Optional[Series[int]]
    id_column: Series[str] = pa.Field(coerce=True)


class Dataset(BaseModel):

    raw_data: DataFrame[DatasetSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_sequences(self) -> List[AnnotatedSequence]:

        data_copy = self.raw_data.copy(deep=True) 

        # ensure that the events are strings
        data_copy[DatasetSchema.event_column] = data_copy[DatasetSchema.event_column].astype(str)

        # since we are interested in calculating confidence of rules
        # we need at least a sequence with length 2
        # as a first step, we can remove all sequences with length 1
        sequence_nunique = data_copy.groupby([DatasetSchema.id_column])[DatasetSchema.event_column].nunique() != 1
        sequence_nunique.name = 'to_keep'
        data_copy = data_copy.merge(sequence_nunique, left_on=[DatasetSchema.id_column], right_index=True)
        data_copy = data_copy[data_copy['to_keep'] == True]

        data_copy_grouped = data_copy.groupby([DatasetSchema.id_column])

        sequences = data_copy_grouped[DatasetSchema.event_column].apply(list)

        result = []

        if DatasetSchema.class_column not in data_copy.columns:

            for index, value in sequences.items():

                result.append(AnnotatedSequence(
                    id_value=index[0][0],
                    sequence_values=value
                    ))

        else:

            classes = data_copy_grouped[DatasetSchema.class_column].apply(list)

            for (index, value), class_value in zip(sequences.items(), classes):
                result.append(AnnotatedSequence(
                    id_value=index[0][0],
                    sequence_values=value,
                    class_value=class_value[0]
                    ))

        return result

    @classmethod
    def from_observations(cls, sequences: List[List[str]], classes: Optional[List[int]] = None):

        # assert length
        if classes is not None:
            assert len(sequences) == len(classes), f"sequences and classes should have the same length"

        # create dataframe entry for each element in database
        records = []

        for id, sequence in enumerate(sequences):
            for e_idx, event in enumerate(sequence):

                record = {
                    'id_column': str(id),
                    'time_column': e_idx,
                    'event_column': event,
                }

                if classes is not None:
                    record['class_column'] = classes[id]

                records.append(record)
                
        raw_data = pd.DataFrame.from_records(records)

        prefix_df = Dataset(
            raw_data=raw_data
        )

        return prefix_df


class DatasetRulesSchema(pa.DataFrameModel):

    antecedent: Series[List[str]]
    consequent: Series[List[str]]

    support: Series[int]
    support_pos: Series[int]
    support_neg: Series[int]

    confidence: Series[float]
    confidence_pos: Series[float]
    confidence_neg: Series[float]

    delta_confidence: Series[float]
    inverse_entropy: Series[float]


class DatasetRules(BaseModel):

    data: DataFrame[DatasetRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create_from_frequent_pattern(cls, freq_pattern: List[FrequentPatternWithConfidence]) -> "DatasetRules":

        # enrich with delta confidence and inverse entropy
        data = pd.DataFrame([{
            **el.model_dump(),
            **{
                DatasetRulesSchema.delta_confidence: el.delta_confidence,
                DatasetRulesSchema.inverse_entropy: el.inverse_entropy
            }
        } for el in freq_pattern])

        return DatasetRules(data=data)


class DatasetUniqueRulesSchema(pa.DataFrameModel):
    id_column: Series[str]
    delta_confidence: Series[List[float]]
    inverse_entropy: Series[List[float]]


class DatasetUniqueRules(BaseModel):
    data: DataFrame[DatasetUniqueRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

