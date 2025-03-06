import pandas as pd
import numpy as np
import pandera as pa

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal, Tuple
from pandera.typing import DataFrame, Series

from src.util.datasets import DatasetSchema
from src.util.custom_types import AnnotatedSequence
from src.preprocess.util.types import AnnotatedSequence, FrequentPatternWithConfidence, BootstrapRound


@pa.extensions.register_check_method(statistics=["min_value", "max_value"], check_type="element_wise")
def is_between(list_obj, *, min_value, max_value):
    return all([min_value <= el <= max_value for el in list_obj])

class DatasetAggregatedSchema(pa.DataFrameModel):
    id_column: Series[str] = pa.Field(coerce=True)
    sequence_values: Series[List[str]]
    class_value: Optional[Series[int]]


class DatasetAggregated(BaseModel):
    data: DataFrame[DatasetAggregatedSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_pandas(cls, data: DataFrame[DatasetSchema]) -> "DatasetAggregated":

        # work on copy
        data_copy = data.copy(deep=True)

        data_copy_grouped = data_copy.groupby([DatasetSchema.id_column])

        # since this column is optional, we add it here and set it to None as its default value
        if DatasetSchema.class_column not in data_copy.columns:
            data_copy[DatasetSchema.class_column] = None

        sequences = data_copy_grouped[DatasetSchema.event_column].apply(list)    
        classes = data_copy_grouped[DatasetSchema.class_column].apply(list)
        
        # check if all class values are identical for a its sequence
        if (data_copy_grouped[DatasetSchema.class_column].nunique() > 1).any():
            raise ValueError(f"class values for sequence {index} are not identical")

        annotated_sequence_records = []

        for (index, value), class_value in zip(sequences.items(), classes):

            annotated_sequence_records.append({
                DatasetAggregatedSchema.id_column: index,
                DatasetAggregatedSchema.sequence_values: value,
                DatasetAggregatedSchema.class_value: class_value[0]
            })

        # format to dataframe
        result = cls(
            data=pd.DataFrame.from_records(annotated_sequence_records)
        )

        return result


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
    centered_inverse_entropy: Series[float]
    total_observations: Series[int]


class DatasetRules(BaseModel):

    data: DataFrame[DatasetRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create_from_bootstrap_rounds(cls, bootstrap_rounds: List[BootstrapRound]) -> "DatasetRules":

        records = []

        for round in bootstrap_rounds:

            total_observations = round.n_samples

            for pattern in round.freq_patterns:

                records.append({
                    **pattern.model_dump(),
                    **{
                        DatasetRulesSchema.total_observations: total_observations,
                        DatasetRulesSchema.delta_confidence: pattern.delta_confidence,
                        DatasetRulesSchema.centered_inverse_entropy: pattern.centered_inverse_entropy
                    }
                })

        data = pd.DataFrame.from_records(records)

        return DatasetRules(data=data)


class DatasetUniqueRulesSchema(pa.DataFrameModel):
    id_column: Series[str]
    delta_confidence: Series[List[float]]
    centered_inverse_entropy: Series[List[float]]
    support: Series[List[float]] = pa.Field(is_between={"min_value": 0, "max_value": 1})

    class Config:
        unique=["id_column"]


class DatasetUniqueRules(BaseModel):
    data: DataFrame[DatasetUniqueRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def rank_rules(self, criterion: Literal[DatasetRulesSchema.delta_confidence, DatasetRulesSchema.centered_inverse_entropy], ascending: bool = False, weighted_by_support: bool = False) -> List[Tuple[str, float]]:

        # collect ids and values
        ids, values = [], []

        # loop through different rows
        for _, row in self.data.iterrows():

            ids.append(row[DatasetUniqueRulesSchema.id_column])

            value = np.array(row[criterion])

            if weighted_by_support:
                support = np.array(row[DatasetUniqueRulesSchema.support])
                value = value * support

            mean_value = np.mean(value)

            values.append(mean_value)

        # convert to dataframe for easier manipulation
        values_df = pd.DataFrame({
            DatasetUniqueRulesSchema.id_column: ids,
            criterion: values
        })

        # add absolute value to sort by
        sorting_feature = f'{criterion}_abs'
        values_df[sorting_feature] = values_df[criterion].abs()

        # sort
        values_df.sort_values(sorting_feature, ascending=ascending, inplace=True)

        return list(zip(values_df[DatasetUniqueRulesSchema.id_column], values_df[criterion]))

