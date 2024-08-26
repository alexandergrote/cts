from .v1.ts_features import *

import pandas as pd
import pandera as pa
import sys
from collections import defaultdict
from datetime import datetime
from pydantic import  BaseModel, field_validator, conint, confloat
from pydantic.config import ConfigDict
from typing import List, Tuple, Dict, Literal
from pandera.typing import DataFrame, Series


class AnnotatedSequence(BaseModel):
    id_value: str
    sequence_values: List[str]
    class_value: int


class FrequentPattern(BaseModel):

    sequence_values: List[str]

    support: int
    support_pos: int
    support_neg: int

    @property
    def inverse_entropy(self) -> float:

        entropy = EntropyCalculator.calculate_entropy(
            probability= self.support_pos / self.support
        )

        return 1 - entropy


class FrequentPatternWithConfidence(BaseModel):

    antecedent: List[str]
    consequent: List[str]

    support: int
    support_pos: int
    support_neg: int

    confidence: float
    confidence_pos: float
    confidence_neg: float

    @property
    def delta_confidence(self) -> float:
        return self.confidence_pos - self.confidence_neg
    
    @property
    def inverse_entropy(self) -> float:

        entropy = EntropyCalculator.calculate_entropy(
            probability= self.support_pos / self.support
        )

        return 1 - entropy


class ConfidenceCalculator:

    @staticmethod
    def calculate_confidence(support_antecedent: int, support_antecedent_and_consequent: int) -> float:
        
        assert support_antecedent >= support_antecedent_and_consequent, f"support antecedent {support_antecedent} should be greater or equal to support antecedent and consequent {support_antecedent_and_consequent}"
        assert support_antecedent >= 0, f"support antecedent {support_antecedent} should be greater than 0"

        if support_antecedent == 0:
            return 0

        return support_antecedent_and_consequent / support_antecedent


class EntropyCalculator:

    @staticmethod
    def calculate_entropy(probability: float) -> float:

        if probability == 0 or probability == 1:
            return 0

        return -probability * np.log2(probability) - (1-probability) * np.log2(1-probability)


class StackObject(BaseModel):
    database: List[List[str]]
    classes: List[int]
    prefix: List[str]

    @classmethod
    def from_annotated_sequences(cls, annotated_sequences: List[AnnotatedSequence], prefix: List[str]):
        
        database = []
        classes = []

        for annotated_sequence in annotated_sequences:
            database.append(annotated_sequence.sequence_values)
            classes.append(annotated_sequence.class_value)
        
        return cls(prefix=prefix, database=database, classes=classes)


class DatasetSchema(pa.DataFrameModel):
    event_column: Series[str] = pa.Field(coerce=True)
    time_column: Series[datetime]
    class_column: Series[int]
    id_column: Series[str]


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
        classes = data_copy_grouped[DatasetSchema.class_column].apply(list)

        result = []

        for (index, value), class_value in zip(sequences.items(), classes):
            result.append(AnnotatedSequence(
                id_value=index[0][0], 
                sequence_values=value, 
                class_value=class_value[0]
                ))

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
    inverse_entropy: Series[float]


class DatasetRules(BaseModel):

    data: DataFrame[DatasetRulesSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create_from_frequent_pattern(cls, freq_pattern: List[FrequentPatternWithConfidence]) -> "DatasetProcessed":

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


class RuleEncoder(BaseModel):

    @staticmethod
    def is_subsequence(subseq, seq):
        it = iter(seq)
        return all(item in it for item in subseq)
    
    def encode(rules: List[List[str]], sequences2classify: List[List[str]], string_separator: str = '_') -> pd.DataFrame:

        # result data
        data = []
        indices = []

        for sequence in sequences2classify:

            row_data = {}
            indices.append(string_separator.join(sequence))

            for rule in rules:

                row_data[f'{string_separator.join(rule)}'] = RuleEncoder.is_subsequence(subseq=rule, seq=sequence)

            
            data.append(row_data)

        df = pd.DataFrame.from_records(data)
        df.index = indices

        return df 


class PrefixSpanNew(BaseModel):
    
    max_sequence_length: conint(ge=0) = sys.maxsize
    min_support_abs: conint(ge=0) = 0
    min_support_rel: confloat(ge=0.0) = 0.0

    @field_validator('min_support_abs', mode='before')
    def _convert_none_to_number(cls, v: Optional[int]):
        if v is None:
            return 0
        return v

    def get_item_counts(self, database: List[List[str]], classes: List[int]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:

        assert len(database) == len(classes)

        freq_items = defaultdict(int)
        freq_items_pos = defaultdict(int)
        freq_items_neg = defaultdict(int)

        # Count support for each item in the projected database
        for sequence, class_value in zip(database, classes):

            used = set()

            for item in sequence:

                if item in used:
                    continue

                freq_items[item] += 1

                if class_value == 0:
                    freq_items_neg[item] += 1

                elif class_value == 1:
                    freq_items_pos[item] += 1
                
                else:
                    raise ValueError(f"Class value {class_value} is not supported")

                used.add(item)

        return freq_items, freq_items_neg, freq_items_pos

    def get_frequent_patterns(self, sequences: List[AnnotatedSequence]) -> List[FrequentPattern]:

        # prepare data for while loop
        # more memory efficient than recursion - prevents stack overflow
        stack = [StackObject.from_annotated_sequences(prefix=[], annotated_sequences=sequences)]
        frequent_pattern_lookup = []
        patterns = []

        while stack:

            stack_object = stack.pop()

            counts, counts_neg, counts_pos = self.get_item_counts(
                stack_object.database,
                stack_object.classes
            )

            for item, count in counts.items():

                if count < self.min_support_abs:
                    continue

                if len(stack_object.prefix) + 1 > self.max_sequence_length:
                    continue

                new_prefix = stack_object.prefix + [item]

                frequent_pattern = FrequentPattern(
                    sequence_values=new_prefix,
                    support=count,
                    support_pos=counts_pos.get(item, 0),
                    support_neg=counts_neg.get(item, 0),
                )

                patterns.append(frequent_pattern)

                new_projected_db = []
                new_classes = []

                for sequence, class_value in zip(stack_object.database, stack_object.classes):

                    try:

                        index = sequence.index(item)
                        new_projected_db.append(sequence[index + 1:])
                        new_classes.append(class_value)

                    except ValueError:
                        continue

                # if the new projected db does not have enough entries
                # that could satisfy the min support threshold
                # continue with next iteration
                if len(new_projected_db) < self.min_support_abs:
                    continue

                stack.append(StackObject(database=new_projected_db, classes=new_classes, prefix=new_prefix))
                frequent_pattern_lookup.append((counts, counts_neg, counts_pos))

        return patterns

    def get_frequent_patterns_with_confidence(self, frequent_patterns: List[FrequentPattern]) -> List[FrequentPatternWithConfidence]:

        lookup = {str(pattern.sequence_values): pattern for pattern in frequent_patterns}
        
        rules = []

        for sequence in frequent_patterns:

            if len(sequence.sequence_values) == 1:
                continue

            for i in range(len(sequence.sequence_values)):

                antecedent = sequence.sequence_values[:i + 1]
                consequent = sequence.sequence_values[i + 1:]

                if len(consequent) == 0:
                    continue

                antecedent_pattern = lookup[str(antecedent)]

                confidence = ConfidenceCalculator.calculate_confidence(
                    support_antecedent=antecedent_pattern.support,
                    support_antecedent_and_consequent=sequence.support
                )

                confidence_pos = ConfidenceCalculator.calculate_confidence(
                    support_antecedent=antecedent_pattern.support_pos,
                    support_antecedent_and_consequent=sequence.support_pos
                )

                confidence_neg = ConfidenceCalculator.calculate_confidence(
                    support_antecedent=antecedent_pattern.support_neg,
                    support_antecedent_and_consequent=sequence.support_neg
                )

                rules.append(
                    FrequentPatternWithConfidence(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=sequence.support,
                        support_pos=sequence.support_pos,
                        support_neg=sequence.support_neg,
                        confidence=confidence,
                        confidence_pos=confidence_pos,
                        confidence_neg=confidence_neg,
                    )
                )

        return rules

    def execute(self, dataset: pd.DataFrame) -> List[FrequentPatternWithConfidence]:

        prefix_df = Dataset(
            raw_data=dataset
        )

        sequences = prefix_df.get_sequences()

        frequent_patterns = self.get_frequent_patterns(sequences)

        frequent_patterns_with_confidence = self.get_frequent_patterns_with_confidence(
            frequent_patterns
        )

        return frequent_patterns_with_confidence


class SPMFeatureSelectorNew(BaseModel, BaseFeatureEncoder):

    prefixspan_config: dict

    criterion: Literal[
        DatasetUniqueRulesSchema.delta_confidence, 
        DatasetUniqueRulesSchema.inverse_entropy
    ] = DatasetUniqueRulesSchema.delta_confidence

    bootstrap_repetitions: int = 5
    bootstrap_sampling_fraction: float = 0.8

    multitesting: Optional[dict] = None
    p_value_threshold: float = 0.01

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

        for train_index, _ in generator:
            mask = data[DatasetSchema.id_column].isin(data_unique.iloc[train_index][DatasetSchema.id_column])

        return data[mask]

    @pa.check_types
    def _bootstrap(self, *, data: DataFrame[DatasetSchema], **kwargs) -> List[FrequentPatternWithConfidence]:

        prefix: PrefixSpanNew = DynamicImport.import_class_from_dict(
            dictionary=self.prefixspan_config
        )

        predictions = []

        for i in tqdm(range(self.bootstrap_repetitions)):

            # select sample
            data_sub = data.groupby(DatasetSchema.class_column, group_keys=False).apply(
                lambda x: self._bootstrap_id_selection(data=x, random_state=i)
            )

            # get output on sample
            prediction = prefix.execute(dataset=data_sub)
    
            predictions.extend(prediction)

        return predictions

    def _get_unique_patterns(self, patterns: List[FrequentPatternWithConfidence]) -> DatasetRules:

        # this consists of two processes:
        # 1) joining bootstrap results
        # 2) joining rules that are essentially the same based on the event order, but differ in their antecedent and consequent

        patterns_df = DatasetRules.create_from_frequent_pattern(
            freq_pattern=patterns
        ).data

        patterns_df[DatasetSchema.id_column] = \
            patterns_df[DatasetRulesSchema.antecedent].astype('str') + \
            patterns_df[DatasetRulesSchema.consequent].astype('str')

        predictions_grouped = patterns_df.groupby([DatasetSchema.id_column]) \
            [[DatasetRulesSchema.delta_confidence, DatasetRulesSchema.inverse_entropy]] \
                .agg(list)

        predictions_grouped.reset_index(inplace=True)

        predictions_grouped[DatasetSchema.id_column] = \
            predictions_grouped[DatasetSchema.id_column].apply(
                lambda x: x.replace('][', ', ').replace('[', '').replace(']', ''))

        unique_predictions = predictions_grouped.groupby(DatasetSchema.id_column).agg('sum')
        unique_predictions.reset_index(inplace=True)

        data = DatasetUniqueRules(
            data=unique_predictions
        )

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
            values = np.zeros_like(obs)  
                
            test = mannwhitneyu(obs, values, alternative='greater')
            p_values.append(test.pvalue)

        p_values_array = np.array(p_values)

        if self.multitesting is None:

            # exclude rules based on p value
            mask = p_values_array < self.p_value_threshold

        else:

            _, pvals_corrected, _, _ = multipletests(p_values_array, **self.multitesting)

            mask = np.array(pvals_corrected) < self.p_value_threshold

        result = DatasetUniqueRules(
            data=data_copy[mask]
        )

        return result


    def _encode_train(self, *args, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        # bootstrap rules
        console.log(f"{self.__class__.__name__}: Bootstrapped Rule Mining")
        patterns = self._bootstrap(data=data_copy)

        # get unique rules
        console.log(f"{self.__class__.__name__}: Obtaining unique rules")
        unique_patterns = self._get_unique_patterns(patterns=patterns)

        # select informative rules
        console.log(f"{self.__class__.__name__}: Selecting rules")
        selected_patterns = self._select_significant_greater_than_zero(data=unique_patterns)

        # encode rules as a binary feature
        console.log(f"{self.__class__.__name__}: Encoding rules")

        sequences = Dataset(
            raw_data=data
        ).get_sequences()

        sequences_values = [el.sequence_values for el in sequences]

        selected_patterns.data['id_column'] = selected_patterns.data['id_column'].apply(lambda x: x.replace("'", '').split(', '))

        encoded_dataframe = RuleEncoder.encode(
            rules=selected_patterns.data['id_column'].to_list(), 
            sequences2classify=sequences_values
        )

        return encoded_dataframe
    
    def _encode_test(self, *args, data: pd.DataFrame, **kwargs):

        assert 'rules' in kwargs, "Rules must be provided to the feature selector"

        # work on copy
        data_copy = data.copy(deep=True)

        # apply rules
        event_sequences_per_id = self._apply_rule_to_ts(rules=kwargs['rules'], event=data_copy)

        return {'data': event_sequences_per_id}
