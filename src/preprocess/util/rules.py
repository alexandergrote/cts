import pandas as pd

from pydantic import BaseModel
from typing import Any, Optional, List


class Rule(BaseModel):

    """

    ('A' -> 'B') -> 'C'

    """

    antecedent: Any
    consequent: str

    # metrics
    support: Optional[float] = None
    confidence: Optional[float] = None


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
