import pandas as pd
import numpy as np
import pickle
import hashlib

from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu
from typing import List, Optional, Iterable
from pydantic import BaseModel, validator, model_validator
from itertools import chain, combinations, product
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

from src.preprocess.base import BaseFeatureEncoder
from src.preprocess.util.rules import Rule
from src.util.constants import RuleFields, Directory
from src.util.custom_logging import console, Pickler, log_time
from src.util.dynamic_import import DynamicImport
from src.util.caching import pickle_cache, PickleCacheHandler, hash_dataframe


class PrefixSpan(BaseModel):

    id_columns: List[str]
    event_column: str
    splitting_symbol: str = ' --> '
    itertool_threshold: Optional[int] = None

    min_support_abs: Optional[int] = None
    min_support_rel: Optional[float] = None

    def _get_combinations(self, sequence: List[str]):

        final_combinations = set()

        window_size = len(sequence)
        if self.itertool_threshold:
            window_size = min(window_size, self.itertool_threshold)

        for idx, _ in enumerate(sequence):
            target_sequence = sequence[idx:window_size + idx]
            target_sequence_combinations = chain.from_iterable(
                [combinations(target_sequence, r) for r in range(1, window_size + 1)])
            final_combinations.update(set(target_sequence_combinations))

        return list(final_combinations)

    @pickle_cache(ignore_caching=False, cachedir='support_dict')
    def _get_support_dict(self, sequences: Iterable[List[str]]) -> dict:

        # additionally, we need to keep track of the support of each sequence
        # so we can calculate confidence of rules
        # we can use a dictionary to store the support of each sequence
        # the key of the dictionary is the sequence
        # the value of the dictionary is the support of the sequence
        sequence_supports = {}

        # we start by creating a list of all possible sequences
        # this list will contain duplicate sequences
        # but we need those for calculating support of each unique sequence
        for sequence in tqdm(sequences, total=len(sequences)):

            # get combinations in sequence
            # itertool combinations returns " n! / r! / (n-r)! " elements
            # this is computationally demanding
            sequence_combinations = self._get_combinations(sequence)

            # to avoid duplicates, keep track of added elements
            added_elements = []

            for sequence_combination in sequence_combinations:

                # calculate support for rule combination
                key = self.splitting_symbol.join(sequence_combination)

                # check if key appears multiple times in sequence combination
                if key in added_elements:
                    continue

                # init count if sequence not already registered
                if key not in sequence_supports:
                    sequence_supports[key] = 1
                    added_elements.append(key)
                    continue

                # increment count
                sequence_supports[key] += 1
                added_elements.append(key)

        return sequence_supports

    @pickle_cache(ignore_caching=False, cachedir='rules')
    def _get_rules(self, sequence_supports: dict):
        
        # to calculate the confidence of rules, we need to iterate over the sequence_supports dictionary
        rule_confidences = {}
        rules = []

        for sequence, support in tqdm(sequence_supports.items(), total=len(sequence_supports)):

            # we only need to consider sequences with length greater than 1
            # since we are interested in rules
            if self.splitting_symbol in sequence:

                # we split the sequence into antecedent and consequent
                # antecedent is the first item of the sequence
                # consequent is the remaining items of the sequence
                sequence_split = sequence.split(self.splitting_symbol)

                for i in range(1, len(sequence_split)):
                    antecedent = sequence_split[:i]
                    consequent = sequence_split[i:]

                    # we calculate the confidence of the rule
                    confidence = support / sequence_supports[self.splitting_symbol.join(antecedent)]

                    # we store the confidence of the rule in the nested rule_confidences dictionary
                    if self.splitting_symbol.join(antecedent) not in rule_confidences:
                        rule_confidences[self.splitting_symbol.join(antecedent)] = {}

                    rule_confidences[self.splitting_symbol.join(antecedent)][self.splitting_symbol.join(consequent)] = confidence

                    rule = Rule(
                        antecedent=self.splitting_symbol.join(antecedent),
                        precedent=self.splitting_symbol.join(consequent),
                        support=support,
                        confidence=confidence
                    )
                    rules.append(rule)

        return rules

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy(deep=True)

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

        # get support of each sequence and store in dict
        sequence_supports = self._get_support_dict(sequences)

        rules = self._get_rules(sequence_supports)

        result = pd.DataFrame.from_records([rule.dict() for rule in rules])

        if self.min_support_abs is not None:
            result = result[result['support'] >= self.min_support_abs]

        if self.min_support_rel is not None:
            tmp = result['support'] / len(data_copy[self.id_columns[0]].unique())
            result = result[tmp >= self.min_support_rel]

        return result


class RuleClassifier(BaseModel):

    rule: List[str]
    _cache: Optional[dict] = None

    def _get_cache_filename(self) -> str:

        filename_components = [self.__class__.__name__]
        filename_components += [str(arg) for arg in self.rule]
        filename_verbose = "__".join(filename_components)

        # create hash for shorter filenames
        hash_object = hashlib.sha1(str.encode(filename_verbose))
        filename_pickle = f"{hash_object.hexdigest()}.pickle"

        return filename_pickle

    def _get_cache_filepath(self) -> Path:
        directory = Directory.CACHING_DIR / 'rule_clf'
        directory.mkdir(exist_ok=True, parents=True)
        filename = self._get_cache_filename()
        return directory / filename
    
    def _write_cache(self):
        cache_file = self._get_cache_filepath()
        with open(cache_file, 'wb') as cachehandle:
            pickle.dump(self._cache, cachehandle)

    @model_validator(mode='after')
    def _init_cache(self):

        if self._cache is None:
            return

        cache_file = self._get_cache_filepath()

        if cache_file.exists():
            with open(cache_file, 'rb') as cachehandle:
                self._cache = pickle.load(cachehandle)

    @staticmethod
    def _check_if_sorted_ascending(x: np.ndarray) -> bool:
        return all(a <= b for a, b in zip(x, x[1:]))
    
    def _apply_rule(self, rule, sequence: List[str]) -> bool:

        sequence_set = set(sequence)
        rule_set = set(rule)

        # rule is not contained in dataset
        if len(rule_set.difference(sequence_set)) != 0:
            return False

        # if elements are present in rule, check if time constraint is met
        # 1) get all possible indices for each rule element in the sequence
        # 2) check if at least one index combination is sorted in an ascending fashion
        # 3) if it does, return true, else temporal constraints of rule are not met

        # get all possible index location for each rule element
        args = [np.argwhere(np.array(sequence) == el).flatten() for el in self.rule]

        # check for all combinations if the indices are sorted
        for combination in product(*args):

            # indices should not be identical, hence referring to the same array element
            # in other words, each combination should have only unique index values
            if len(np.unique(combination)) < len(combination):
                continue

            # if they are sorted, return true
            if self._check_if_sorted_ascending(combination):
                return True

        return False

    def _apply_rule_from_cache(self, sequence: List[str]) -> bool:

        """
        Checks if rule is contained in a sequence

        Args:
            sequence: list of events

        Returns:
            boolean indicator if rule is present in sequence

        """

        # check if sequence is already in cache
        sequence_str = "__".join(sequence)
        if sequence_str in self._cache:
            return self._cache[sequence_str]
        else:
            result = self._apply_rule(rule=self.rule, sequence=sequence)
            self._cache[sequence_str] = result
            return result 

    def apply_rules(self, sequences: List[List[str]]) -> List[bool]:

        if self._cache is None:
            result = [self._apply_rule(rule=self.rule, sequence=sequence) for sequence in sequences]
        else:

            result = [self._apply_rule_from_cache(sequence=sequence) for sequence in sequences]
            self._write_cache()

        return result


class CausalRuleFeatureSelector(BaseModel, BaseFeatureEncoder):

    treatment_attr_name: str
    treatment_attr_value: str

    prefixspan_config: dict

    splitting_symbol: str
    repetitions: int = 5
    bootstrap_sampling_fraction: float = 0.8
    p_value_threshold: float = 0.01
    support_threshold: float = 0.001
    abs_support_threshold: int = 100
    conf_threshold: float = 0.05

    # ts dataset specific variables
    ts_id_columns: List[str]
    ts_event_column: str
    ts_datetime_column: str

    multitesting: Optional[dict] = None
    n_features: Optional[int] = None

    @validator("multitesting")
    def _set_multitesting(cls, v):

        if v is None:
            return v

        if len(v.keys()) == 0:
            return None

        return v

    @log_time(key='rule_mining')
    def _rule_mining(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy(deep=True)

        prefix = DynamicImport.import_class_from_dict(
            dictionary=self.prefixspan_config
        )

        # divide datafarme into control and treatment rules
        mask_treatment = data_copy[self.treatment_attr_name] == int(self.treatment_attr_value)

        # check masking
        assert sum(mask_treatment) < len(data_copy)
        assert sum(mask_treatment) > 0

        rules_control_df = prefix.execute(data=data_copy[~mask_treatment], **kwargs)
        
        rules_treatment_df = prefix.execute(data=data_copy[mask_treatment], **kwargs)

        # calculate delta
        index_col = [RuleFields.ANTECEDENT.value, RuleFields.PRECEDEDENT.value]
        rules = rules_control_df.merge(right=rules_treatment_df, left_on=index_col, right_on=index_col, how='outer')

        # if rule does not exist in one or another dataframe, fill existing values
        missing_value_mapping = [
            (RuleFields.SUPPORT, 0),
            (RuleFields.CONFIDENCE, 0)
        ]

        for rule_field, value in missing_value_mapping:
            columns = rules.filter(like=rule_field.value).columns
            rules.loc[:, columns] = rules[columns].fillna(value)

        # init ranking column
        rules[RuleFields.RANKING.value] = 1

        num_sequences = data[self.ts_id_columns[0]].nunique()

        for attribute in [RuleFields.SUPPORT, RuleFields.CONFIDENCE]:

            string_control = f"{attribute.value}_x"
            string_treatment = f"{attribute.value}_y"

            if attribute == RuleFields.CONFIDENCE:
                rules[attribute.value] = rules[string_treatment] - rules[string_control]
            elif attribute == RuleFields.SUPPORT:
                rules[attribute.value] = (rules[string_treatment] + rules[string_control]) / num_sequences
            else:
                raise ValueError(f"Attribute values does not match expectations")

            rules[RuleFields.RANKING.value] *= rules[attribute.value].abs()
            mask_support = rules[RuleFields.SUPPORT.value] * num_sequences >= self.abs_support_threshold

        return rules[mask_support]

    @staticmethod
    def is_subsequence(seq1, seq2):

        assert len(seq1) > len(seq2), "seq1 should be longer than seq2"

        j = 0  # Pointer for seq2
        # Iterate over seq1
        for i in range(len(seq1)):
            
            if j == len(seq2):  # Check if all elements of seq2 are found
                break

            if seq1[i] == seq2[j]:  # If current element matches, move to next element in seq2
                j += 1

        return j == len(seq2)  # Check if all elements in seq2 were found

    def _check_stat_significance_of_subsequences(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        # work on copy
        data_copy = data.copy(deep=True)

        # get rule lengths
        rule_length_column = 'rule_length'
        data_copy[rule_length_column] = data_copy['index'].apply(lambda x: len(x.split(self.splitting_symbol)))

        # get unique rule lengths
        rule_lengths = data_copy[rule_length_column].unique()

        # sort rule lengths in ascending order
        rule_lengths.sort()

        # loop through rule lengths
        for rule_length in rule_lengths:

            # get all rules with the current rule length
            mask = data_copy[rule_length_column] == rule_length
            rules_short = data_copy[mask]

            # get all rules that 1 longer than the current rule length
            mask = data_copy[rule_length_column] == rule_length + 1
            rules_long = data_copy[mask]

            if rules_long.empty:
                continue

            # rules to drop
            rules_to_drop = []

            # loop through all rules with the current rule length
            for _, row_short in rules_short.iterrows():

                # compare with all rules that are 1 longer
                for _, row_long in rules_long.iterrows():

                    # check if subsequence
                    rule_short = list(filter(None, row_short['index'].split(self.splitting_symbol)))
                    rule_long = list(filter(None, row_long['index'].split(self.splitting_symbol)))
                    
                    if not self.is_subsequence(rule_long, rule_short):
                        continue

                    # check difference in delta confidence
                    expresssion = f"{RuleFields.CONFIDENCE.value}_"
                    confidence_short = row_short.filter(like=expresssion).values
                    confidence_long = row_long.filter(like=expresssion).values

                    # get delta confidence of both rules
                    """_, p_value = ttest_rel(confidence_short, confidence_long)

                    if all(confidence_short == confidence_long):

                        rules_to_drop.append(row_long['index'])
                        continue"""

                    _, p_value = wilcoxon(confidence_short, confidence_long)
                    
                    # if there is no significant difference, it is safe to drop the rule
                    # if there is a significant difference, we keep the rule
                    # it may be that the longer rule 
                    if p_value > self.p_value_threshold:
                        rules_to_drop.append(row_long['index'])

        # drop rules
        mask = data_copy['index'].isin(rules_to_drop)
        
        data_copy = data_copy[~mask]

        return data_copy

    def _get_unique_rules(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
            
        # work on copy
        rules = data.copy(deep=True)

        # init column names
        rule_column = 'index'
        sorting_column = 'abs_avg_confidence'
        confidence_column = 'mean_confidence'

        # sort by abs delta confidence to make drop duplicates easier
        # keep the rules with the higher absolute delta confidence
        rules[sorting_column] = rules[confidence_column].abs()
        rules.sort_values(by=sorting_column, ascending=False, inplace=True)

        # prepare rules for vectorized comparison
        rules_as_list = rules[rule_column].str.split(self.splitting_symbol)
        rules_as_str = rules_as_list.apply(lambda x: f'{self.splitting_symbol}'.join(list(filter(None, x))))

        # drop duplicates
        mask = rules_as_str.duplicated(keep='first')
        rules = rules[~mask]

        # drop sorting column
        rules.drop(columns=[sorting_column], inplace=True)

        return rules

    @pickle_cache(ignore_caching=True)
    def _select_significant_greater_than_zero(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """
        Select rules based on their ranking

        Args:
            data: pd.DataFrame containing all rules
            **kwargs:

        Returns:

        """

        # work on copy
        result = data.copy(deep=True)

        # check data validity
        num_columns: int = None
        for field in [RuleFields.SUPPORT, RuleFields.CONFIDENCE, RuleFields.RANKING]:
            columns = result.filter(like=field.value).columns
            if num_columns is None:
                num_columns = len(columns)
                assert num_columns > 0
            assert num_columns == len(columns)

        # reformat result to test for significance
        result_as_list = result.filter(like=RuleFields.CONFIDENCE.value+'_').apply(lambda x: x.to_list(), axis=1)

        # init p value column
        p_value_column_name = 'p_value'
        p_value_corrected_column_name = 'p_value_corrected'
        result[p_value_column_name] = None
        p_value_column_idx = result.columns.get_loc(p_value_column_name)

        for idx, el in enumerate(result_as_list):

                
            if all(np.array(el) == 0):
                p_value = 1.0
            else:
                y = np.zeros_like(el) + self.conf_threshold
                x = np.abs(el)
                
                test = mannwhitneyu(x, y, alternative='greater')
                p_value = test.pvalue

            

            result.iloc[idx, p_value_column_idx] = p_value

        if self.multitesting is None:

            # exclude rules based on p value
            mask = result[p_value_column_name] < self.p_value_threshold

            result = result[mask]

            return result

        _, pvals_corrected, _, _ = multipletests(result[p_value_column_name], **self.multitesting)

        result[p_value_corrected_column_name] = pvals_corrected

        mask = result[p_value_corrected_column_name] < self.p_value_threshold
        result = result[mask]

        return result

    def _bootstrap_id_selection(self, data: pd.DataFrame, random_state: int) -> pd.DataFrame:

        assert len(self.ts_id_columns) == 1, "Only one id column is supported"
        id_column = self.ts_id_columns[0]

        data_unique = data[self.ts_id_columns + [self.treatment_attr_name]].drop_duplicates()
        
        assert data_unique[self.ts_id_columns].duplicated().sum() == 0, "id-target-mapping should be unique"
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-self.bootstrap_sampling_fraction, random_state=random_state)
        generator = sss.split(data_unique[self.ts_id_columns].values.reshape(-1,), data_unique[self.treatment_attr_name].values)

        for train_index, _ in generator:
            mask = data[id_column].isin(data_unique.iloc[train_index][id_column])

        return data[mask]

    @pickle_cache(ignore_caching=True)
    def _bootstrap(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        result = None

        for i in tqdm(range(self.repetitions)):

            # select sample
            data_sub = data.groupby(self.treatment_attr_name, group_keys=False).apply(
                lambda x: self._bootstrap_id_selection(data=x, random_state=i)
            )

            # get output on sample
            conf_prediction = self._rule_mining(data=data_sub, **kwargs)

            # format output
            index_column = 'index'
            analysis_columns = [RuleFields.SUPPORT.value, RuleFields.CONFIDENCE.value, RuleFields.RANKING.value]
            conf_prediction[index_column] = conf_prediction[RuleFields.ANTECEDENT.value] + self.splitting_symbol*2 + conf_prediction[RuleFields.PRECEDEDENT.value]
            df = conf_prediction.set_index(index_column)[analysis_columns]
            df.columns = [f"{col}_{i}" for col in analysis_columns]

            # merge with existing results
            # avoid for loop in doing so
            if result is None:
                result = df
                continue

            result = result.merge(df, left_index=True, right_index=True)

        support_columns = result.filter(like=RuleFields.SUPPORT.value).columns
        result = result[result[support_columns].mean(axis=1) > self.support_threshold]

        result.reset_index(inplace=True)

        return result

    def _apply_rule_to_ts(self, *, rules: pd.DataFrame, event: pd.DataFrame, **kwargs):

        hash_str = '__'.join([hash_dataframe(rules), hash_dataframe(event)])

        cache_handler = PickleCacheHandler(
            filepath=Path('rule_clf') / f'{hash_str}.pickle'
        )

        res = cache_handler.read()

        if res is not None:
            return res


        # work on copy
        rules_copy = rules.copy(deep=True)
        event_copy = event.copy(deep=True)

        # check data validity
        assert 'index' in rules_copy.columns
        assert self.ts_datetime_column in event_copy.columns
        assert self.ts_event_column in event_copy.columns
        for col in self.ts_id_columns:
            assert col in event_copy.columns

        # to apply the rules, we need to make sure that the incoming dataframe is ordered
        event_copy = event_copy.sort_values(by=self.ts_datetime_column)
        event_copy.reset_index(drop=True, inplace=True)

        # create sequence for each id
        column_name = 'event_sequences'
        event_grouped = event_copy.groupby(self.ts_id_columns)
        event_sequences = event_grouped.apply(lambda x:  list(x[self.ts_event_column]))
        event_sequences.name = column_name
        event_sequences_df = event_sequences.to_frame()

        dataframes = [event_sequences_df.reset_index()]

        for _, row in tqdm(rules_copy.iterrows(), total=len(rules_copy)):

            rule = list(filter(None, row['index'].split(self.splitting_symbol)))
            rule_clf = RuleClassifier(rule=rule, _cache={})
            
            sequences = event_sequences_df[column_name].to_list()
            result = rule_clf.apply_rules(sequences)

            dataframe = pd.DataFrame({row['index']: result})

            dataframes.append(dataframe)

        event_sequences_df = pd.concat(dataframes, axis=1)

        # fill missing values
        event_sequences_df.fillna(False, inplace=True)

        # drop column name
        event_sequences_df.drop(column_name, axis=1, inplace=True)
        event_sequences_df.reset_index(inplace=True)

        data_class = event_copy[self.ts_id_columns +  [self.treatment_attr_name]].drop_duplicates()
        event_sequences_per_id = event_sequences_df.merge(
            data_class, 
            right_on=self.ts_id_columns, 
            left_on=self.ts_id_columns
        )          
        
        # drop id columns
        event_sequences_per_id.drop(columns= self.ts_id_columns +['index'], inplace=True, errors='ignore')

        # write to cache
        cache_handler.write(event_sequences_per_id)

        return event_sequences_per_id

    def _encode_train(self, *args, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        # bootstrap rules
        console.log(f"{self.__class__.__name__}: Bootstrapped Rule Mining")
        rules = self._bootstrap(data=data_copy, **kwargs)
        console.log(f"{len(rules)} rules")
        rules_logging_dict = {'0_bootstrapped': rules}

        for column in [RuleFields.SUPPORT.value, RuleFields.CONFIDENCE.value, RuleFields.RANKING.value]:

            # calculate aggregated values
            rules_filtered = rules.filter(like=column)

            for measure in ['mean', 'std']:
                series = getattr(rules_filtered, measure)(axis=1)
                series.name = f'{measure}_{column}'
                rules = rules.merge(series, left_index=True, right_index=True)

        # select unique rules
        console.log(f"{self.__class__.__name__}: Selecting unique rules")
        rules = self._get_unique_rules(data=rules)
        console.log(f"{len(rules)} rules")
        rules_logging_dict['1_unique_rules'] = rules

        # select only statistically significant rules
        console.log(f"{self.__class__.__name__}: Excluding statistically insignificant rules")
        rules = self._select_significant_greater_than_zero(data=rules)
        
        console.log(f"{len(rules)} rules")
        rules_logging_dict['1_significant_greater'] = rules

        Pickler.write(rules_logging_dict, 'rules_logging.pickle')

        # applying rules
        console.log(f"{self.__class__.__name__}: Applying rules to time series")
        event_sequences_per_id = self._apply_rule_to_ts(rules=rules, event=data_copy)

        """Paper Analysis Start"""

        from IPython import embed; embed()

        columns = [col for col in event_sequences_per_id.columns if '-->' in col]

        result = {}

        for column in columns:
            result[column] = event_sequences_per_id[event_sequences_per_id[column]][self.treatment_attr_name].mean() - event_sequences_per_id[self.treatment_attr_name].mean()

        result = pd.Series(result)
        result.name = 'deviation_from_mean_target'

        result = rules.merge(result, left_on='index', right_index=True)

        Pickler.write(result, "rules_conf_target.pickle")

        """Paper Analysis End"""
        
        # save output
        kwargs['rules'] = rules
        kwargs['data'] = event_sequences_per_id

        return kwargs

    def _encode_test(self, *args, data: pd.DataFrame, **kwargs):

        assert 'rules' in kwargs, "Rules must be provided to the feature selector"

        # work on copy
        data_copy = data.copy(deep=True)

        # apply rules
        event_sequences_per_id = self._apply_rule_to_ts(rules=kwargs['rules'], event=data_copy)

        return {'data': event_sequences_per_id}
