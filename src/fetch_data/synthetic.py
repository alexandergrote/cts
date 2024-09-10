import numpy as np
import pandas as pd
from collections import Counter
from pydantic import BaseModel, model_validator
from typing import List, Dict

from src.fetch_data.base import BaseDataLoader
from src.preprocess.util.rules import RuleClassifier
from src.util.caching import pickle_cache
from src.util.constants import Directory
from src.util.datasets import Dataset, DatasetSchema

rng = np.random.default_rng(seed=0)


class DataLoader(BaseModel, BaseDataLoader):

    configuration: Dict[str, Dict[str, float]]
    n_samples: int

    sequence_elements: List[str]
    separator: str

    # config columns
    select_key: str = 'selection'
    class_key: str = 'class'
    configuration_result_column: str = 'configuration'

    @model_validator(mode='after')
    def check_configuration(self):

        # check if columns are present
        for key in self.configuration:
            if self.select_key not in self.configuration[key] or self.class_key not in self.configuration[key]:
                raise ValueError(f'Configuration for {key} is not complete')
            
        # check if selection probabilities are greater than 0 and smaller than 1
        for key in self.configuration:
            if not 0 <= self.configuration[key][self.select_key] <= 1:
                raise ValueError(f'Probability for {key} is not in [0, 1]')
            
        # check if class probabilities are greater than 0 and smaller than 1
        for key in self.configuration:
            if not 0 <= self.configuration[key][self.class_key] <= 1:
                raise ValueError(f'Probability for {key} is not in [0, 1]')
            
        # check if sum of selection probabilities is greater equal 0 but smaller equal 1
        for key in [self.select_key]:
            if not 0 <= sum([self.configuration[k][key] for k in self.configuration]) <= 1:
                raise ValueError(f'Sum of probabilities is not in [0, 1]')


    @pickle_cache(ignore_caching=False, cachedir=Directory.CACHING_DIR / 'synthetic')
    def get_data(self) -> Dataset:

        sequence_goals = {k: self.configuration[k][self.select_key] * self.n_samples for k in self.configuration}
        sequence_goals[None] = self.n_samples - sum(sequence_goals.values())

        # list of elements that will be drawn randomly if they are not part of the configuration
        result = []

        # draw classes for sampled elements
        classes = []

        # configuration list
        configuration_results = []

        default_class_probability = 0.5

        # draw elements according to their weights
        while len(result) < self.n_samples:

            random_rule_length = rng.integers(low=2, high=len(self.sequence_elements), size=1)[0]
            random_rule = rng.choice(self.sequence_elements, size=random_rule_length, replace=False)
            random_rule_str = self.separator.join(random_rule)

            # is drawn rule contained in the configuration?
            classifiers = [RuleClassifier(rule=key.split(self.separator)) for key in self.configuration.keys()]
            configuration_indicator = [clf.apply_rules(sequences=[random_rule])[0] for clf in classifiers]
            num_rules = sum(configuration_indicator)

            # exclude potential conflicting rules
            if num_rules > 1:
                continue

            configuration_result = list(self.configuration.keys())[np.argmax(configuration_indicator)] if num_rules == 1 else None

            counter = Counter(configuration_results)
            
            if sequence_goals[configuration_result] - counter.get(configuration_result, 0) <= 0:
                continue

            class_probability = self.configuration[configuration_result][self.class_key] if configuration_result is not None else default_class_probability
            class_result = rng.choice([0, 1], p=[1 - class_probability, class_probability])


            result.append(random_rule_str)
            classes.append(class_result)
            configuration_results.append(configuration_result)


        # summarize data in a pandas dataframe
        data = pd.DataFrame({
            DatasetSchema.event_column: result,
            DatasetSchema.class_column: classes,
            self.configuration_result_column: configuration_results
        })

        data.fillna('None', inplace=True)

        # calculate the relative frequency of each element
        relative_frequencies = data[self.configuration_result_column].value_counts(normalize=True).sort_values(ascending=False)

        # calculate the probability of class 1 for each element
        probabilities = data.groupby(self.configuration_result_column)[DatasetSchema.class_column].mean().sort_values(ascending=False)


        print('--relative frequencies--')
        print(relative_frequencies)

        print('--probabilities--')
        print(probabilities)

        print('--data--')
        records = []

        for idx, (rule, target) in enumerate(zip(data[DatasetSchema.event_column], data[DatasetSchema.class_column])):

            time_idx = 0
            for rule_el in rule.split(self.separator):
                records.append({DatasetSchema.id_column: idx, DatasetSchema.time_column: time_idx, DatasetSchema.event_column: rule_el, DatasetSchema.class_column: target})
                time_idx += 1


        data = pd.DataFrame.from_records(records)

        return data


if __name__ == '__main__':

    from src.util.config_import import YamlConfigLoader

    config = YamlConfigLoader.read_yaml(key='fetch_data')

    data_loader = DataLoader(**config['params'])
    data = data_loader.execute()['data']

    event_column = '%event_column%'
    id_column = '%id_column%'
    time_column = '%time_column%'
    target_column = '%target_column%'

    # unique events
    print('unique events')
    print(data[event_column].nunique())

    # get unique event counts
    print('event counts')
    print(data[event_column].value_counts(normalize=False).mean())

    # get average sequence length
    print('sequence length')
    print(data.groupby(id_column).size().mean())

    # get min sequence length
    print('min sequence length')
    print(data.groupby(id_column).size().min())

    # get max sequence length
    print('max sequence length')
    print(data.groupby(id_column).size().max())

    # get average class distribution
    print('class distribution')
    print(data['%target_column%'].mean())

    