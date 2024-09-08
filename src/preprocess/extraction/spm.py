import pandas as pd
import sys

from collections import defaultdict
from pandera.typing import DataFrame
from pydantic import BaseModel, conint, ConfigDict, field_validator
from typing import List, Optional, Union, Tuple, Dict


from src.preprocess.base import BaseFeatureEncoder
from src.preprocess.util.rules import RuleEncoder
from src.preprocess.util.metrics import ConfidenceCalculator
from src.preprocess.util.types import AnnotatedSequence, FrequentPattern, StackObject, FrequentPatternWithConfidence
from src.preprocess.util.datasets import DatasetRulesSchema, DatasetUniqueRulesSchema
from src.util.datasets import Dataset, DatasetSchema


class PrefixSpan(BaseModel, BaseFeatureEncoder):
    
    max_sequence_length: conint(ge=0) = sys.maxsize
    min_support_abs: conint(ge=0) = 0

    model_config = ConfigDict(extra="forbid")

    @field_validator('min_support_abs', mode='before')
    def _convert_none_to_number(cls, v: Optional[int]):
        if v is None:
            return 0
        return v

    def get_item_counts(self, database: List[List[str]], classes: Union[List[int], List[None]]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:

        freq_items = defaultdict(int)
        freq_items_pos = defaultdict(int)
        freq_items_neg = defaultdict(int)

        assert len(database) == len(classes)

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

                used.add(item)

        return freq_items, freq_items_neg, freq_items_pos

    def get_frequent_patterns(self, sequences: List[AnnotatedSequence]) -> List[FrequentPattern]:

        # prepare data for while loop
        # more memory efficient than recursion - prevents stack overflow
        stack = [StackObject.from_annotated_sequences(prefix=[], annotated_sequences=sequences)]

        patterns = []

        while stack:

            stack_object = stack.pop()

            counts, counts_neg, counts_pos = self.get_item_counts(
                stack_object.database,
                stack_object.classes
            )

            # check if classes have been provided
            class_provided: bool = stack_object.classes[0] is not None

            for item, count in counts.items():

                if count < self.min_support_abs:
                    continue

                if len(stack_object.prefix) + 1 > self.max_sequence_length:
                    continue

                new_prefix = stack_object.prefix + [item]

                frequent_pattern = FrequentPattern(
                    sequence_values=new_prefix,
                    support=count,
                    support_pos=counts_pos.get(item, 0) if class_provided else None,
                    support_neg=counts_neg.get(item, 0) if class_provided else None,
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
                if len(new_projected_db) + len(new_prefix) < self.min_support_abs:
                    continue

                stack.append(StackObject(database=new_projected_db, classes=new_classes, prefix=new_prefix))

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

                try:

                    antecedent_pattern = lookup[str(antecedent)]

                except KeyError as e:

                    raise KeyError(f"""Lookup dict does not contain all the necessary values. This is likely due to an incomplete list of frequent pattners. Missing: {e}""")

                confidence = ConfidenceCalculator.calculate_confidence(
                    support_antecedent=antecedent_pattern.support,
                    support_antecedent_and_consequent=sequence.support
                )

                confidence_pos, confidence_neg = None, None

                if antecedent_pattern.support_pos is not None:

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

    def summarise_patterns_in_dataframe(self, data: DataFrame[DatasetSchema]) -> pd.DataFrame:

        frequent_patterns_with_confidence = self.execute(
            dataset=data
        )

        df = pd.DataFrame([el.model_dump() for el in frequent_patterns_with_confidence])

        if DatasetRulesSchema.support_pos not in df.columns:
            raise ValueError(f"""Column {DatasetRulesSchema.support_pos} not found in dataframe""")

        if df[DatasetRulesSchema.support_pos].isna().all() == False:
            df[DatasetUniqueRulesSchema.delta_confidence] = df[DatasetRulesSchema.confidence_pos] - df[DatasetRulesSchema.confidence_neg]
            df.sort_values(by=[DatasetUniqueRulesSchema.delta_confidence], inplace=True, ascending=False)

        return df.dropna(axis=1)

    def _encode_train(self, data: pd.DataFrame, **kwargs) -> Dict:

        # work on copy
        data_copy = data.copy(deep=True)

        prefix_df = Dataset(
            raw_data=data_copy
        )

        sequences = prefix_df.get_sequences()
        sequence_values = [el.sequence_values for el in sequences]
        class_values = [el.class_value for el in sequences]

        frequent_patterns = self.get_frequent_patterns(sequences)

        rules = [el.sequence_values for el in frequent_patterns]

        encoded_dataframe = RuleEncoder.encode(
            rules=rules, 
            sequences2classify=sequence_values
        )

        encoded_dataframe[DatasetSchema.class_column] = class_values

        kwargs['data'] = encoded_dataframe
        kwargs['rules'] = rules

        return kwargs
    
    def _encode_test(self, *, data: pd.DataFrame, **kwargs) -> Dict:
        # encode rules as a binary feature on test data

        # assert requirements
        assert 'rules' in kwargs, "Rules must be provided to the feature selector"
        rules = kwargs['rules']

        assert isinstance(rules, list), "Rules must be of type list"
        assert all(isinstance(el, list) for el in rules), "Rules must be of type list of lists"

        # assert type of data
        assert isinstance(data, pd.DataFrame), "Data must be of type pd.DataFrame"
        
        data_copy = Dataset(
            raw_data=data.copy(deep=True)
        )

        sequences = data_copy.get_sequences()

        sequences_values = [el.sequence_values for el in sequences]
        class_values = [el.class_value for el in sequences]

        encoded_dataframe = RuleEncoder.encode(
            rules=rules, 
            sequences2classify=sequences_values
        )

        encoded_dataframe[DatasetSchema.class_column] = class_values
        
        return {'data': encoded_dataframe}

    def execute(self, dataset: pd.DataFrame) -> List[FrequentPatternWithConfidence]:

        prefix_df = Dataset(
            raw_data=dataset
        )

        sequences = prefix_df.get_sequences()

        frequent_patterns = self.get_frequent_patterns(sequences)

        # todo: this step can be optimised if we ignore the possibility of different antecedent and consequent
        frequent_patterns_with_confidence = self.get_frequent_patterns_with_confidence(
            frequent_patterns
        )

        return frequent_patterns_with_confidence
