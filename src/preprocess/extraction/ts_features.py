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


class DatasetProcessedSchema(pa.DataFrameModel):

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


class DatasetProcessed(BaseModel):

    data: DataFrame[DatasetProcessedSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create_from_frequent_pattern(cls, freq_pattern: List[FrequentPatternWithConfidence]) -> "DatasetProcessed":

        # enrich with delta confidence and inverse entropy
        data = pd.DataFrame([{
            **el.model_dump(),
            **{
                DatasetProcessedSchema.delta_confidence: el.delta_confidence,
                DatasetProcessedSchema.inverse_entropy: el.inverse_entropy
            }
        } for el in freq_pattern])

        return DatasetProcessed(data=data)

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

    criterion: Literal[DatasetProcessedSchema.delta_confidence, DatasetProcessedSchema.inverse_entropy] = DatasetProcessedSchema.delta_confidence

    bootstrap_repetitions: int = 5
    bootstrap_sampling_fraction: float = 0.8

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

    def _encode_train(self, *args, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        # bootstrap rules
        console.log(f"{self.__class__.__name__}: Bootstrapped Rule Mining")

        patterns = self._bootstrap(data=data_copy)

        patterns_df = DatasetProcessed.create_from_frequent_pattern(
            freq_pattern=patterns
        ).data

        patterns_df[DatasetSchema.id_column] = \
            patterns_df[DatasetProcessedSchema.antecedent].astype('str') + \
            patterns_df[DatasetProcessedSchema.consequent].astype('str')

        predictions_grouped = patterns_df.groupby([DatasetSchema.id_column]) \
            [self.criterion].agg(['mean', 'std'])

        predictions_grouped.reset_index(inplace=True)

        # get unique rules
        console.log(f"{self.__class__.__name__}: Obtaining unique rules")

        from IPython import embed; embed()

        predictions_grouped[DatasetSchema.id_column] = \
            predictions_grouped[DatasetSchema.id_column].apply(
                lambda x: x.replace('][', ', ').replace('[', '').replace(']', ''))



        return predictions_grouped
    
    def _encode_test(self, *args, data: pd.DataFrame, **kwargs):
        return super()._encode_test(*args, **kwargs)

