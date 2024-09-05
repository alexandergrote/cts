import pandas as pd

from pydantic import BaseModel
from typing import List


class RuleClassifier(BaseModel):

    rule: List[str]

    def apply_rules(self, sequences: List[List[str]]) -> List[bool]:

        result = []

        for sequence in sequences:
            result.append(
                RuleEncoder.is_subsequence(self.rule, sequence)
            )

        return result


class RuleEncoder(BaseModel):

    @staticmethod
    def is_subsequence(subseq: List[str], seq: List[str]):
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
