from .v1.ts_features import *

import pandas as pd
import pandera as pa
import sys
from collections import defaultdict
from datetime import datetime
from pydantic import  BaseModel, field_validator, conint, confloat
from pydantic.config import ConfigDict
from typing import List, Tuple, Dict
from pandera.typing import DataFrame, Series

from src.preprocess.util.rules import Rule


class AnnotatedSequence(BaseModel):
    id_value: str
    sequence_values: List[str]
    class_value: int

class FrequentPattern(BaseModel):
    sequence_values: List[str]

    support: int
    support_pos: int
    support_neg: int

    # will only be calculated for sequences with length > 2
    confidence: Optional[float] = None  
    confidence_pos: Optional[float] = None
    confidence_neg: Optional[float] = None
    delta_confidence: Optional[float] = None

class ConfidenceCalculator:

    @staticmethod
    def calculate_confidence(support_antecedent: int, support_antecedent_and_consequent: int) -> float:
        
        assert support_antecedent >= support_antecedent_and_consequent, f"support antecedent {support_antecedent} should be greater or equal to support antecedent and consequent {support_antecedent_and_consequent}"
        assert support_antecedent > 0, f"support antecedent {support_antecedent} should be greater than 0"

        return support_antecedent_and_consequent / support_antecedent

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

class PrefixSpanDatasetSchema(pa.DataFrameModel):
    event_column: Series[str] = pa.Field(coerce=True)
    time_column: Series[datetime]
    class_column: Series[int]
    id_column: Series[str]

class PrefixSpanDataset(BaseModel):

    raw_data: DataFrame[PrefixSpanDatasetSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_sequences(self) -> List[AnnotatedSequence]:

        data_copy = self.raw_data.copy(deep=True) 

        # ensure that the events are strings
        data_copy[PrefixSpanDatasetSchema.event_column] = data_copy[PrefixSpanDatasetSchema.event_column].astype(str)

        # since we are interested in calculating confidence of rules
        # we need at least a sequence with length 2
        # as a first step, we can remove all sequences with length 1
        sequence_nunique = data_copy.groupby([PrefixSpanDatasetSchema.id_column])[PrefixSpanDatasetSchema.event_column].nunique() != 1
        sequence_nunique.name = 'to_keep'
        data_copy = data_copy.merge(sequence_nunique, left_on=[PrefixSpanDatasetSchema.id_column], right_index=True)
        data_copy = data_copy[data_copy['to_keep'] == True]

        data_copy_grouped = data_copy.groupby([PrefixSpanDatasetSchema.id_column])

        sequences = data_copy_grouped[PrefixSpanDatasetSchema.event_column].apply(list)
        classes = data_copy_grouped[PrefixSpanDatasetSchema.class_column].apply(list)

        result = []

        for (index, value), class_value in zip(sequences.items(), classes):
            result.append(AnnotatedSequence(
                id_value=index[0][0], 
                sequence_values=value, 
                class_value=class_value[0]
                ))

        return result
    
class PrefixSpanNew(BaseModel):

    rule_symbol: str = " --> "
    
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

            lookup_exist: bool = False
            if len(frequent_pattern_lookup) > 0:
                lookup_exist = True
                counts_old, counts_old_neg, counts_old_pos = frequent_pattern_lookup.pop()

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

                confidence, confidence_pos, confidence_neg = None, None, None
                confidence_difference = None

                if lookup_exist:

                    confidence = ConfidenceCalculator.calculate_confidence(
                        support_antecedent=counts_old[item],
                        support_antecedent_and_consequent=counts[item]
                    )

                    confidence_pos = ConfidenceCalculator.calculate_confidence(
                        support_antecedent=counts_old_pos[item],
                        support_antecedent_and_consequent=counts_pos[item]
                    )

                    confidence_neg = ConfidenceCalculator.calculate_confidence(
                        support_antecedent=counts_old_neg[item],
                        support_antecedent_and_consequent=counts_neg[item]
                    )

                    confidence_difference = confidence_pos - confidence_neg

                frequent_pattern = FrequentPattern(
                    sequence_values=new_prefix,
                    support=count,
                    support_pos=counts_pos.get(item, 0),
                    support_neg=counts_neg.get(item, 0),
                    confidence=confidence,
                    confidence_pos=confidence_pos,
                    confidence_neg=confidence_neg,
                    delta_confidence=confidence_difference
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

    def execute(self, dataset: pd.DataFrame) -> List[AnnotatedSequence]:

        prefix_df = PrefixSpanDataset(
            raw_data=dataset
        )

        sequences = prefix_df.get_sequences()

        frequent_patterns = self.get_frequent_patterns(sequences)

        print(frequent_patterns)

        return frequent_patterns




