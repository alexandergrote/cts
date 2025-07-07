import pandas as pd
import numpy as np
import pandera as pa

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal, Tuple
from pandera.typing import DataFrame, Series

from src.util.datasets import DatasetSchema
from src.preprocess.util.subgroup_measures import compute_chi_square, compute_fisher, compute_phi_coefficient, compute_leverage
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
            data_copy[DatasetSchema.class_conlumn] = None

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
    chi_squared_p_values: Series[float]
    fisher_odds_ratio: Series[float]
    fisher_odds_ratio_p_values: Series[float]
    entropy: Series[float]
    phi: Series[float]
    leverage: Series[float]

    total_observations: Series[int]


class DatasetRules(BaseModel):

    data: DataFrame[DatasetRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create_from_bootstrap_rounds(cls, bootstrap_rounds: List[BootstrapRound], skip_interesting_measures: bool = False) -> "DatasetRules":

        records = []

        for b_round in bootstrap_rounds:

            total_observations = b_round.n_samples
            total_observations_neg = b_round.n_samples_neg
            total_observations_pos = b_round.n_samples_pos

            for pattern in b_round.freq_patterns:

                # calculation of other interesting measures takes time and is not strictly needed for cost-benefit analysis
                if skip_interesting_measures:

                    records.append({
                        **pattern.model_dump(),
                        **{
                            DatasetRulesSchema.total_observations: total_observations,
                            DatasetRulesSchema.delta_confidence: pattern.delta_confidence,
                            DatasetRulesSchema.centered_inverse_entropy: pattern.centered_inverse_entropy,
                            DatasetRulesSchema.entropy: pattern.entropy,
                            DatasetRulesSchema.chi_squared: -999.0,
                            DatasetRulesSchema.chi_squared_p_values: -999.0,
                            DatasetRulesSchema.phi: -999.0, 
                            DatasetRulesSchema.fisher_odds_ratio: -999.0,
                            DatasetRulesSchema.fisher_odds_ratio_p_values: -999.0,
                            DatasetRulesSChema.leverage: -999.0,

                        }
                    })

                    continue

                chi2, chi2_p = compute_chi_square(
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

                phi = compute_phi_coefficient(
                    rule_pos=pattern.support_pos,
                    rule_neg=pattern.support_neg,
                    non_rule_pos=total_observations_pos - pattern.support_pos,
                    non_rule_neg=total_observations_neg - pattern.support_neg,
                )

                leverage = compute_leverage(
                    num_observations=total_observations, 
                    num_pos=total_observations_pos, 
                    num_seq=pattern.support_pos + pattern.support_neg, 
                    num_pos_seq=pattern.support_pos
                )

                records.append({
                    **pattern.model_dump(),
                    **{
                        DatasetRulesSchema.total_observations: total_observations,
                        DatasetRulesSchema.delta_confidence: pattern.delta_confidence,
                        DatasetRulesSchema.centered_inverse_entropy: pattern.centered_inverse_entropy,
                        DatasetRulesSchema.entropy: pattern.entropy,
                        DatasetRulesSchema.leverage: leverage,
                        DatasetRulesSchema.chi_squared: chi2,
                        DatasetRulesSchema.chi_squared_p_values: chi2_p,
                        DatasetRulesSchema.phi: phi,
                        DatasetRulesSchema.fisher_odds_ratio: fisher_odds,
                        DatasetRulesSchema.fisher_odds_ratio_p_values: fisher_p
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
    entropy: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    phi: Series[object] = pa.Field(is_list_of_floats=pa.Check.is_list_of_floats)
    support: Series[object] = pa.Field(is_between={"min_value": 0, "max_value": 1})

    class Config:
        unique=["id_column"]


class DatasetUniqueRules(BaseModel):
    
    data: DataFrame[DatasetUniqueRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def rank_rules(self, criterion: Literal[DatasetRulesSchema.delta_confidence, DatasetRulesSchema.centered_inverse_entropy, DatasetRulesSchema.fisher_odds_ratio], ascending: bool = False, weighted_by_support: bool = False) -> List[Tuple[str, float]]:

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


class DatasetDeltaConfidencePValuesSchema(pa.DataFrameModel):
    id_column: Series[str]
    
    p_values: Series[float]
    corrected_p_values: Series[float]
    mask: Series[bool]  # mask for the p-values that are significant

    class Config:
        unique=["id_column"]


class DatasetDeltaConfidencePValues(BaseModel):

    data: DataFrame[DatasetDeltaConfidencePValuesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create_from_unique_rules(cls, data: DatasetUniqueRules, pvalues: np.ndarray, corrected_pvalues: np.ndarray, mask: np.ndarray):

        df = pd.DataFrame({
            DatasetDeltaConfidencePValuesSchema.id_column: data.data[DatasetUniqueRulesSchema.id_column],
            DatasetDeltaConfidencePValuesSchema.p_values: pvalues,
            DatasetDeltaConfidencePValuesSchema.corrected_p_values: corrected_pvalues,
            DatasetDeltaConfidencePValuesSchema.mask: mask
        })

        return cls(data=df)