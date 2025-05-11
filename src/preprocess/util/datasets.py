import pandas as pd
import numpy as np
import pandera as pa

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal, Tuple
from pandera.typing import DataFrame, Series
from scipy.stats import chi2_contingency, fisher_exact

from src.util.datasets import DatasetSchema
from src.preprocess.util.types import BootstrapRound


@pa.extensions.register_check_method(statistics=["min_value", "max_value"], check_type="element_wise")
def is_between(list_obj, *, min_value, max_value):
    return all([min_value <= el <= max_value for el in list_obj])


@pa.extensions.register_check_method(statistics=[], check_type="element_wise")
def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(x, str) for x in lst)

@pa.extensions.register_check_method(statistics=[], check_type="element_wise")
def is_list_of_floats(lst):
    return isinstance(lst, list) and all(isinstance(x, float) for x in lst)


def compute_chi_square(rule_pos: int, rule_neg: int, non_rule_pos: int, non_rule_neg: int) -> Tuple[float, float]:
    
    contingency_matrix = np.array([[rule_pos, rule_neg], [non_rule_pos, non_rule_neg]])

    try:
        chi2, p, _, _ = chi2_contingency(contingency_matrix)
    except ValueError:
        chi2, p = float('nan'), float('nan')  # if any issue (e.g., division by zero)

    return chi2, p


def compute_fisher(rule_pos: int, rule_neg: int, non_rule_pos: int, non_rule_neg: int) -> Tuple[float, float]:
    """
    Compute Fisher's exact test for a 2x2 contingency table.
    
    Args:
        rule_pos: Number of positive cases where the rule applies
        rule_neg: Number of negative cases where the rule applies
        non_rule_pos: Number of positive cases where the rule does not apply
        non_rule_neg: Number of negative cases where the rule does not apply
        
    Returns:
        Tuple containing the odds ratio and p-value
    """
    contingency_matrix = np.array([[rule_pos, rule_neg], [non_rule_pos, non_rule_neg]])
    
    try:
        odds_ratio, p_value = fisher_exact(contingency_matrix)
    except ValueError:
        odds_ratio, p_value = float('nan'), float('nan')  # if any issue occurs
        
    return odds_ratio, p_value


class DatasetAggregatedSchema(pa.DataFrameModel):
    id_column: Series[str] = pa.Field(coerce=True)
    sequence_values: Series[object] = pa.Field(is_list_of_strings=pa.Check.is_list_of_strings)
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
        sequences_with_multiple_classes = data_copy_grouped[DatasetSchema.class_column].nunique() > 1
        if sequences_with_multiple_classes.any():
            problematic_sequences = sequences_with_multiple_classes[sequences_with_multiple_classes].index.tolist()
            raise ValueError(f"class values for sequences {problematic_sequences} are not identical")

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

    antecedent: Series[object] = pa.Field(is_list_of_strings=pa.Check.is_list_of_strings)
    consequent: Series[object] = pa.Field(is_list_of_strings=pa.Check.is_list_of_strings)

    support: Series[int]
    support_pos: Series[int]
    support_neg: Series[int]

    confidence: Series[float]
    confidence_pos: Series[float]
    confidence_neg: Series[float]

    delta_confidence: Series[float]
    centered_inverse_entropy: Series[float]
    chi_squared: Series[float]
    fisher_odds_ratio: Series[float]
    fisher_p_value: Series[float]
    entropy: Series[float]

    total_observations: Series[int]


class DatasetRules(BaseModel):

    data: DataFrame[DatasetRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create_from_bootstrap_rounds(cls, bootstrap_rounds: List[BootstrapRound]) -> "DatasetRules":

        records = []

        for b_round in bootstrap_rounds:

            total_observations = b_round.n_samples
            total_observations_neg = b_round.n_samples_neg
            total_observations_pos = b_round.n_samples_pos

            for pattern in b_round.freq_patterns:

                chi2, p = compute_chi_square(
                    rule_pos=pattern.support_pos,
                    rule_neg=pattern.support_neg,
                    non_rule_pos=total_observations_pos - pattern.support_pos,
                    non_rule_neg=total_observations_neg - pattern.support_neg,
                )
                
                fisher_odds, fisher_p = compute_fisher(
                    rule_pos=pattern.support_pos,
                    rule_neg=pattern.support_neg,
                    non_rule_pos=total_observations_pos - pattern.support_pos,
                    non_rule_neg=total_observations_neg - pattern.support_neg,
                )

                records.append({
                    **pattern.model_dump(),
                    **{
                        DatasetRulesSchema.total_observations: total_observations,
                        DatasetRulesSchema.delta_confidence: pattern.delta_confidence,
                        DatasetRulesSchema.centered_inverse_entropy: pattern.centered_inverse_entropy,
                        DatasetRulesSchema.entropy: pattern.entropy,
                        DatasetRulesSchema.chi_squared: chi2,
                        DatasetRulesSchema.fisher_odds_ratio: fisher_odds,
                        DatasetRulesSchema.fisher_p_value: fisher_p,
                    }
                })

        data = pd.DataFrame.from_records(records)

        return DatasetRules(data=data)


class DatasetUniqueRulesSchema(pa.DataFrameModel):
    id_column: Series[str]
    delta_confidence: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    centered_inverse_entropy: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    chi_squared: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    fisher_odds_ratio: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    fisher_p_value: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    entropy: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    support: Series[object] = pa.Field(is_between={"min_value": 0, "max_value": 1})

    class Config:
        unique=["id_column"]


class DatasetUniqueRules(BaseModel):
    
    data: DataFrame[DatasetUniqueRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def rank_rules(self, criterion: Literal[DatasetRulesSchema.delta_confidence, DatasetRulesSchema.centered_inverse_entropy, DatasetRulesSchema.fisher_odds_ratio, DatasetRulesSchema.fisher_p_value], ascending: bool = False, weighted_by_support: bool = False) -> List[Tuple[str, float]]:

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

