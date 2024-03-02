import numpy as np
import pandas as pd

from string import ascii_lowercase
from typing import List, Optional, Tuple, Union, Generator
from pydantic import BaseModel, model_validator, field_validator
from itertools import combinations, chain, count, product
from sklearn.datasets import make_classification

from src.fetch_data.base import BaseDataLoader, BaseDataset
from src.preprocess.ts_feature_selection import RuleClassifier
from src.util.dynamic_import import DynamicImport
from src.util.caching import pickle_cache


class EventGenerator(BaseModel):

    max_events: int

    def get_events(self) -> List[str]:

        counter = 0
        result: List[str] = []

        for size in count(1):
            for s in product(ascii_lowercase, repeat=size):
                event: str = "".join(s)
                result.append(event)
                counter += 1

                if counter >= self.max_events:
                    return result


class SequenceGenerator(BaseModel):

    n_sequences: int
    max_sequence_length: int

    random_seed: int = 42

    sequence_weights_splitting_char: str
    sequence_weights: Optional[dict] = None

    sequences_to_ignore: Optional[List[str]] = None

    def _get_random_sequence_combinations(self, events: List[str]):

        rng = np.random.default_rng(seed=self.random_seed)

        stop = True
        counter_fails = 0

        n_samples = set()
        if self.sequence_weights is not None:
            n_samples = set(self.sequence_weights.keys())

        classifiers = []
        if self.sequences_to_ignore is not None:

            for rule in self.sequences_to_ignore:

                rule = rule.split(self.sequence_weights_splitting_char)

                classifiers.append(
                    RuleClassifier(rule=rule)
                )

        while stop:

            random_rule_length = rng.integers(low=2, high=len(events), size=1)

            random_event_sequence = rng.choice(events, size=random_rule_length)

            if classifiers:
                skip: bool = False
                for clf in classifiers:
                    if clf.apply_rule(sequence=random_event_sequence):
                        skip = True
                        break

                if skip:
                    continue

            random_event_sequence_str = self.sequence_weights_splitting_char.join(random_event_sequence)
            len_n_samples = len(n_samples)
            n_samples.add(random_event_sequence_str)

            # increment counter if
            if len_n_samples == len(n_samples):
                counter_fails += 1

            if counter_fails == max(self.n_sequences, 1000):
                raise ValueError("Sampling failed. Consider increasing the possible value combinations")

            if len(n_samples) == self.n_sequences:
                stop = False

        return [tuple(el.split(self.sequence_weights_splitting_char)) for el in n_samples]

    def _get_temporal_constrained_sequence_combinations(self, events: List[str]):

        sequence_combinations = [combinations(events, r) for r in range(1, self.max_sequence_length + 1)]
        sequence_combinations = list(chain.from_iterable(sequence_combinations))

        return sequence_combinations

    @field_validator("sequence_weights")
    def _check_length(cls, v):

        if v is None:
            return v

        assert len(v.keys()) == 1

        return v

    @staticmethod
    def _get_weight_adjustment(elements: List, target_weight: float, clf: RuleClassifier):

        # occurence frequency
        num_occurence = sum([clf.apply_rule(sequence=list(el)) for el in elements])
        num_non_occurence = len(elements) - num_occurence

        assert num_occurence + num_non_occurence == len(elements)

        # default weight
        default_w = 1 / len(elements)

        numerator = target_weight * default_w * (num_occurence + num_non_occurence) - num_occurence * default_w
        denominator = num_occurence * (1 - target_weight)

        # weight adjustment
        if denominator == 0:
            weight_adjustment = 999999999
        else:
            weight_adjustment = numerator / denominator

        return weight_adjustment

    def _get_weights(self, sequence_combinations: List[Tuple[str]]) -> Optional[np.ndarray]:

        if self.sequence_weights is None:
            return None

        weights = np.ones((len(sequence_combinations))) / len(sequence_combinations)

        # unpack dict
        sequence, target_weight = list(self.sequence_weights.items())[0]

        # check dict contents
        assert self.sequence_weights_splitting_char in sequence

        rule = sequence.split(self.sequence_weights_splitting_char)
        rule_clf = RuleClassifier(rule=rule)

        weight_adjustment = self._get_weight_adjustment(
            elements=sequence_combinations,
            target_weight=target_weight,
            clf=rule_clf
        )

        for idx, sequence_combination in enumerate(sequence_combinations):
            if rule_clf.apply_rule(sequence=list(sequence_combination)):
                weights[idx] += weight_adjustment

        # normalize weights
        weights = weights / weights.sum()

        return weights

    def get_sequences(self, events: List[str]) -> List[str]:

        sequence_combinations = self._get_random_sequence_combinations(events=events)
        #print(sequence_combinations)

        weights = self._get_weights(sequence_combinations=sequence_combinations)

        rng = np.random.default_rng(seed=self.random_seed)
        selected_choices = rng.choice(sequence_combinations, size=self.n_sequences, p=weights)

        rule = 'a_c'.split(self.sequence_weights_splitting_char)
        rule_clf = RuleClassifier(rule=rule)

        print(sum([rule_clf.apply_rule(seq) for seq in selected_choices]) / len(selected_choices))

        return selected_choices


class TimeSeriesDataset(BaseModel, BaseDataset):

    event_generator: Union[dict, EventGenerator]
    sequence_generator: Union[dict, SequenceGenerator]

    id_column: str = 'id_column'
    time_column: str = 'timestamp'
    event_column: str = 'event_column'

    data: Optional[pd.DataFrame] = None
    ids: Optional[List[str]] = None

    random_seed: int = 42

    class Config:
        arbitrary_types_allowed=True

    @field_validator("event_generator", "sequence_generator")
    def _init_classes(cls, v):
        return DynamicImport.import_class_from_dict(v)

    def _get_dataframes(self, sequences: List[str], ids: List[int]):

        assert len(sequences) == len(ids)

        for sequence, id in zip(sequences, ids):

            data = pd.DataFrame({
                self.id_column: id,
                self.time_column: range(len(sequence)),
                self.event_column: sequence
            })

            yield data

    def _get_ids(self, sequences: Optional[List]) -> List:

        ids = list(range(len(sequences)))

        if self.ids is not None:
            rng = np.random.default_rng(seed=self.random_seed)
            ids = rng.choice(self.ids, len(sequences), replace=False)

        return ids

    def get_data(self) -> pd.DataFrame:

        event_list = self.event_generator.get_events()

        sequences = self.sequence_generator.get_sequences(events=event_list)

        ids = self._get_ids(sequences=sequences)

        dataframes = self._get_dataframes(sequences=sequences, ids=ids)

        data = pd.concat(dataframes)

        return data


class StaticDataset(BaseModel, BaseDataset):

    n_samples: int
    random_seed: int

    n_features: int = 20
    n_informative: int = 2
    n_redundant: int = 0
    n_repateted: int = 0

    flip_y: float = 0.1
    weights: Optional[np.ndarray] = None  # class weights

    n_classes: int = 2
    n_clusters_per_class: int = 2

    id_column: str = 'id_column'
    target_column: str = 'target'

    data: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed=True

    @model_validator(mode='after')
    def _validate_n_features(self):
        assert self.n_features >= self.n_informative - self.n_repateted - self.n_redundant
        return self

    def get_data(self) -> pd.DataFrame:

        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_repeated=self.n_redundant,
            n_redundant=self.n_redundant,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=self.weights,
            flip_y=self.flip_y,
            n_classes=self.n_classes,
            shuffle=False
        )

        data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))

        # assign column labels
        base_id = 'static'
        n_other = self.n_features - self.n_informative - self.n_repateted - self.n_redundant
        columns = [f"{base_id}_informative_{i}" for i in range(self.n_informative)]
        columns += [f"{base_id}_redundant_{i}" for i in range(self.n_redundant)]
        columns += [f"{base_id}_repeated_{i}" for i in range(self.n_repateted)]
        columns += [f"{base_id}_other_{i}" for i in range(n_other)]
        columns += [self.target_column]

        data.columns = columns

        data[self.id_column] = range(len(data))

        self.data = data

        return data


class SyntheticDatasets(BaseModel, BaseDataLoader):

    static_dataset: Union[dict, StaticDataset]
    time_series_positive: Union[dict, TimeSeriesDataset]
    time_series_negative: Union[dict, TimeSeriesDataset]

    static_only: bool = False
    time_series_only: bool = False

    class Config:
        arbitrary_types_allowed=True

    @field_validator("static_dataset", "time_series_positive", "time_series_negative")
    def _init_datasets(cls, v):
        return DynamicImport.import_class_from_dict(v)

    @staticmethod
    def _get_time_series_data_with_aligned_index(static_dataset: StaticDataset, ts_datasets: List[TimeSeriesDataset]) -> Generator[pd.DataFrame, None, None]:

        data = static_dataset.data.copy(deep=True)

        if data is None:
            raise ValueError("Data is not set")

        unique_values = data[static_dataset.target_column].unique()
        unique_values.sort()

        for unique_val, ts in zip(unique_values, ts_datasets):

            data_sub = data[data[static_dataset.target_column] == unique_val]
            ids = data_sub[static_dataset.id_column].unique()

            ts.sequence_generator.n_sequences = len(ids)
            ts.ids = ids

            data_ts = ts.get_data()

            yield data_ts

    @pickle_cache(ignore_caching=True)
    def execute(self) -> dict:

        n_observations = self.static_dataset.n_samples
        n_ts_pos = self.time_series_positive.sequence_generator.n_sequences
        n_ts_neg = self.time_series_negative.sequence_generator.n_sequences

        assert n_observations == n_ts_neg + n_ts_pos

        static_data = self.static_dataset.get_data()

        if self.static_only:
            return {'data': static_data}

        ts_data_all = list(self._get_time_series_data_with_aligned_index(
            static_dataset=self.static_dataset,
            ts_datasets=[self.time_series_negative, self.time_series_positive]
        ))

        ts_data = pd.concat(ts_data_all)

        return {'case': static_data, 'event': ts_data}


if __name__ == '__main__':

    ts_data = TimeSeriesDataset(

        event_generator_kwargs={
            'max_events': 10
        },
        sequence_generator_kwargs={
            'n_sequences': 100,
            'max_sequence_length': 3,
            'sequence_weights': {
                'a,b': 0.5
            }
        }
    ).get_data()

    print(ts_data)

    static_data = StaticDataset(
        n_samples=200,
        random_seed=42
    ).get_data()

    print(static_data)





