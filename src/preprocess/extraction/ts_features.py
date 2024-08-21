from .v1.ts_features import *

import pandas as pd
from pydantic import  BaseModel
from pydantic.config import ConfigDict
from typing import List, Iterable, Tuple, Dict
from itertools import chain, combinations, product

from src.preprocess.util.rules import Rule


class Sequence(BaseModel):
    id_values: List[str]
    sequence_values: List[str]


class PrefixSpanDataset(BaseModel):

    id_columns: List[str]
    event_column: str
    raw_data: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_sequences(self) -> List[Sequence]:

        data_copy = self.raw_data.copy(deep=True)

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

        return [Sequence(id_values=list(index), sequence_values=value) for index, value in sequences.items()]
    

class PrefixSpanNew(BaseModel):

    rule_symbol: str = " --> "
    
    window_size: Optional[int] = None
    min_support_abs: Optional[int] = None
    min_support_rel: Optional[float] = None

    def get_combinations(self, sequence: List[str]) -> List[Tuple[str]]:

        final_combinations = set()

        window_size = len(sequence)

        if self.window_size:
            window_size = min(window_size, self.window_size)

        for idx, _ in enumerate(sequence):
            target_sequence = sequence[idx:window_size + idx]
            target_sequence_combinations = chain.from_iterable(
                [combinations(target_sequence, r) for r in range(1, window_size + 1)])
            final_combinations.update(set(target_sequence_combinations))

        return list(final_combinations)

    def get_support(self, sequences: List[Sequence]) -> Dict[str, int]:

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
            sequence_combinations = self.get_combinations(sequence.sequence_values)

            # to avoid duplicates, keep track of added elements
            added_elements = []

            for sequence_combination in sequence_combinations:

                # calculate support for rule combination
                key = self.rule_symbol.join(sequence_combination)

                # check if key appears multiple times in sequence combination
                if key in added_elements:
                    continue

                # init count if sequence not already registered
                if key not in sequence_supports:
                    sequence_supports[key] = 1
                else:
                    # increment count
                    sequence_supports[key] += 1
                
                # add element to the already seen list
                added_elements.append(key)

        return sequence_supports

    def get_rules(self, sequence_supports: dict) -> List[]:
        
        # to calculate the confidence of rules, we need to iterate over the sequence_supports dictionary
        rule_confidences = {}
        rules = []

        for sequence, support in tqdm(sequence_supports.items(), total=len(sequence_supports)):

            # we only need to consider sequences with length greater than 1
            # since we are interested in rules
            if self.rule_symbol not in sequence:
                continue

            # we split the sequence into antecedent and consequent
            # antecedent is the first item of the sequence
            # consequent is the remaining items of the sequence
            sequence_split = sequence.split(self.rule_symbol)

            for i in range(1, len(sequence_split)):
                antecedent = sequence_split[:i]
                consequent = sequence_split[i:]

                # we calculate the confidence of the rule
                confidence = support / sequence_supports[self.rule_symbol.join(antecedent)]

                # we store the confidence of the rule in the nested rule_confidences dictionary
                if self.rule_symbol.join(antecedent) not in rule_confidences:
                    rule_confidences[self.rule_symbol.join(antecedent)] = {}

                rule_confidences[self.rule_symbol.join(antecedent)][self.rule_symbol.join(consequent)] = confidence

                rule = Rule(
                    antecedent=self.rule_symbol.join(antecedent),
                    precedent=self.rule_symbol.join(consequent),
                    support=support,
                    confidence=confidence
                )

                rules.append(rule)

        return rules

    def execute(self, dataset: PrefixSpanDataset) -> List[Sequence]:

        sequences = dataset.get_sequences()

        support_dict = self.get_support(sequences=sequences)

        rules = self.get_rules(sequence_supports=support_dict)



