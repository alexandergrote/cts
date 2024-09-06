import pandas as pd

from pydantic import BaseModel
from typing import List, ClassVar


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

    string_separator: ClassVar[str] = '_'

    @staticmethod
    def is_subsequence(subseq: List[str], seq: List[str]):
        it = iter(seq)
        return all(item in it for item in subseq)
    
    @staticmethod
    def encode_rule_id(rule: List[str]) -> str:
        return RuleEncoder.string_separator.join(rule)
    
    @staticmethod
    def encode(rules: List[List[str]], sequences2classify: List[List[str]], ) -> pd.DataFrame:

        # result data
        data = []
        indices = []

        for sequence in sequences2classify:

            row_data = {}
            idx = RuleEncoder.encode_rule_id(sequence)
            indices.append(idx)

            for rule in rules:
                rule_id = RuleEncoder.encode_rule_id(rule)
                row_data[rule_id] = RuleEncoder.is_subsequence(subseq=rule, seq=sequence)

            data.append(row_data)

        df = pd.DataFrame.from_records(data)
        df.index = indices

        return df 
