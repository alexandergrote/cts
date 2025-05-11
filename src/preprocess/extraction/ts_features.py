import pandas as pd
import numpy as np
import pandera as pa

from pydantic import  BaseModel, field_validator, confloat
from typing import List, Literal, Optional
from pandera.typing import DataFrame
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.util.dynamic_import import DynamicImport
from src.util.custom_logging import console
from src.preprocess.extraction.spm import PrefixSpan
from src.preprocess.util.types import BootstrapRound
from src.preprocess.util.rules import RuleEncoder
from src.util.datasets import Dataset, DatasetSchema
from src.preprocess.util.datasets import DatasetRulesSchema, DatasetRules, DatasetUniqueRules, DatasetUniqueRulesSchema, DatasetAggregated, DatasetAggregatedSchema
from src.preprocess.base import BaseFeatureEncoder
from src.util.caching import environ_pickle_cache


class SPMFeatureSelector(BaseModel, BaseFeatureEncoder):

    prefixspan_config: dict

    criterion: Literal[
        DatasetUniqueRulesSchema.delta_confidence, 
        DatasetUniqueRulesSchema.centered_inverse_entropy
    ] = DatasetUniqueRulesSchema.delta_confidence

    bootstrap_repetitions: int = 5
    bootstrap_sampling_fraction: confloat(ge=0, le=1) = 0.8

    multitesting: Optional[dict] = None
    p_value_threshold: confloat(ge=0, le=1) = 0.01
    criterion_buffer: confloat(ge=0, le=1) = 0

    @field_validator("multitesting", mode="before")
    def _set_multitesting(cls, v):

        if v is None:
            return v

        if len(v.keys()) == 0:
            return None

        return v

    def _bootstrap_id_selection(self, data: pd.DataFrame, random_state: int) -> pd.DataFrame:

        data_unique = data[[DatasetSchema.id_column, DatasetSchema.class_column]].drop_duplicates()
        
        assert data_unique[DatasetSchema.id_column].duplicated().sum() == 0, "id-target-mapping should be unique"
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-self.bootstrap_sampling_fraction, random_state=random_state)
        generator = sss.split(data_unique[DatasetSchema.id_column].values.reshape(-1,), data_unique[DatasetSchema.class_column].values)

        mask = None

        for train_index, _ in generator:
            mask = data[DatasetSchema.id_column].isin(data_unique.iloc[train_index][DatasetSchema.id_column])

        if mask is None:
            raise ValueError("Mask should not be None")

        return data[mask]

    @pa.check_types
    def _bootstrap(self, *, data: DataFrame[DatasetSchema], **kwargs) -> List[BootstrapRound]:

        prefix: PrefixSpan = DynamicImport.import_class_from_dict(
            dictionary=self.prefixspan_config
        )

        predictions = []

        for i in tqdm(range(self.bootstrap_repetitions)):

            # select sample
            data_sub = pd.concat([
                self._bootstrap_id_selection(data=x, random_state=i)
                for _, x in data.groupby(DatasetSchema.class_column, group_keys=False)
            ])

            prefix_df = Dataset(
                raw_data=data_sub
            )

            sequences = prefix_df.get_sequences()
            n_pos = prefix_df.get_number_positives()
            n_neg = prefix_df.get_number_negatives()

            frequent_patterns = prefix.get_frequent_patterns(sequences)

            # todo: this step can be optimised if we ignore the possibility of different antecedent and consequent
            prediction = prefix.get_frequent_patterns_with_confidence(
                frequent_patterns
            )
    
            predictions.append(BootstrapRound(
                n_samples=len(data_sub),
                n_samples_pos=n_pos,
                n_samples_neg=n_neg,
                freq_patterns=prediction
            ))

        return predictions

    def _get_unique_patterns(self, bootstrap_rounds: List[BootstrapRound]) -> DatasetUniqueRules:

        # this consists of two processes:
        # 1) joining bootstrap results
        # 2) joining rules that are essentially the same based on the event order, but differ in their antecedent and consequent

        patterns_df = DatasetRules.create_from_bootstrap_rounds(
            bootstrap_rounds=bootstrap_rounds
        ).data

        patterns_df[DatasetSchema.id_column] = \
            patterns_df[DatasetRulesSchema.antecedent].astype('str') + \
            patterns_df[DatasetRulesSchema.consequent].astype('str')
        
        # normalize support
        # needed for ranking later
        patterns_df[DatasetRulesSchema.support] = patterns_df[DatasetRulesSchema.support] / patterns_df[DatasetRulesSchema.total_observations]

        # columns with multiple values given as a list
        columns_with_lists = [
            DatasetRulesSchema.delta_confidence, 
            DatasetRulesSchema.centered_inverse_entropy,
            DatasetRulesSchema.entropy,
            DatasetRulesSchema.chi_squared,
            DatasetRulesSchema.fisher_odds_ratio,
            DatasetRulesSchema.support
        ]

        predictions_grouped = patterns_df.groupby([DatasetSchema.id_column]) \
            [columns_with_lists].agg(list)

        predictions_grouped.reset_index(inplace=True)

        console.log("Number of unique patterns: {}".format(predictions_grouped.shape[0]))

        predictions_grouped[DatasetSchema.id_column] = \
            predictions_grouped[DatasetSchema.id_column].apply(
                lambda x: x.replace('][', ', ').replace('[', '').replace(']', ''))

        # calculate the average delta_confidence for each row
        avg_criterion_col = f'avg_{self.criterion}'
        predictions_grouped[avg_criterion_col] = predictions_grouped[self.criterion].apply(lambda x: np.mean(np.abs(x)))

        # group by id_column and find the index of the row with the highest average delta_confidence for each group
        idx = predictions_grouped.groupby('id_column')[avg_criterion_col].idxmax()

        # select the rows with the highest average delta_confidence for each group
        unique_predictions = predictions_grouped.loc[idx]
        
        unique_predictions.reset_index(inplace=True)

        data = DatasetUniqueRules(
            data=unique_predictions
        )

        console.log("Number of unique patterns after aggregation: {}".format(data.data.shape[0]))

        return data
    
    def _select_significant_greater_than_zero(self, *, data: DatasetUniqueRules, **kwargs) -> DatasetUniqueRules:

        """
        Conduct statistical tests to select informative rules

        Args:
            data: DatasetRules containing all rules
            **kwargs:

        Returns:

        """

        # work on copy
        data_copy = data.data.copy(deep=True)
        
        # keep track of p values
        p_values = []

        for _, row in data_copy.iterrows():

            # get observations
            obs = np.abs(np.array(row[self.criterion]))
            values = np.zeros_like(obs) + self.criterion_buffer
                
            test = mannwhitneyu(obs, values, alternative='greater')
            p_values.append(test.pvalue)

        p_values_array = np.array(p_values)

        if self.multitesting is None:

            # exclude rules based on p value
            mask = p_values_array < self.p_value_threshold

        else:

            _, pvals_corrected, _, _ = multipletests(p_values_array, **self.multitesting)

            # debugging purposes
            data_copy['p_vals'] = pvals_corrected

            mask = np.array(pvals_corrected) < self.p_value_threshold

        result = DatasetUniqueRules(
            data=data_copy[mask]
        )

        console.log("Number of unique rules after selection: {}".format(result.data.shape[0]))

        return result

    @environ_pickle_cache()
    def _encode_train(self, *args, data: pd.DataFrame, **kwargs) -> dict:

        # work on copy
        data_copy = data.copy(deep=True)

        # bootstrap rules
        console.log(f"{self.__class__.__name__}: Bootstrapped Rule Mining")
        bootsrap_rounds = self._bootstrap(data=data_copy)

        # get unique rules
        console.log(f"{self.__class__.__name__}: Obtaining unique rules")
        unique_patterns = self._get_unique_patterns(bootstrap_rounds=bootsrap_rounds)

        # select informative rules
        console.log(f"{self.__class__.__name__}: Selecting rules")
        selected_patterns = self._select_significant_greater_than_zero(data=unique_patterns)

        # encode rules as a binary feature
        console.log(f"{self.__class__.__name__}: Encoding rules")

        data_agg = DatasetAggregated.from_pandas(
            data=data
        ).data

        class_values = data_agg[DatasetAggregatedSchema.class_value].to_list()
        sequences_values = data_agg[DatasetAggregatedSchema.sequence_values].to_list()

        selected_patterns.data[DatasetUniqueRulesSchema.id_column] = \
            selected_patterns.data[DatasetUniqueRulesSchema.id_column]\
                .apply(lambda x: x.replace("'", '').split(', '))

        encoded_dataframe = RuleEncoder.encode(
            rules=selected_patterns.data[DatasetUniqueRulesSchema.id_column].to_list(), 
            sequences2classify=sequences_values
        )

        encoded_dataframe[DatasetSchema.class_column] = class_values

        # check number of rows
        df_check = data_copy[[DatasetSchema.id_column, DatasetSchema.class_column]].drop_duplicates()
        assert len(df_check) == len(encoded_dataframe), "Number of rows does not match"

        # save output
        kwargs['rules'] = selected_patterns
        kwargs['data'] = encoded_dataframe

        return kwargs
    
    @environ_pickle_cache()
    def _encode_test(self, *args, data: pd.DataFrame, **kwargs) -> dict:

        # encode rules as a binary feature on test data

        # assert requirments
        assert 'rules' in kwargs, "Rules must be provided to the feature selector"
        rules = kwargs['rules']
        assert isinstance(rules, DatasetUniqueRules), "Rules must be of type DatasetRules"

        # assert type of data
        assert isinstance(data, pd.DataFrame), "Data must be of type pd.DataFrame"
        
        data_agg = DatasetAggregated.from_pandas(
            data=data
        ).data

        class_values = data_agg[DatasetAggregatedSchema.class_value].to_list()
        sequences_values = data_agg[DatasetAggregatedSchema.sequence_values].to_list()


        encoded_dataframe = RuleEncoder.encode(
            rules=rules.data[DatasetUniqueRulesSchema.id_column].to_list(), 
            sequences2classify=sequences_values
        )

        encoded_dataframe[DatasetSchema.class_column] = class_values
        
        return {'data': encoded_dataframe}
