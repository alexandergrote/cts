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
from src.util.filepath_converter import FilepathConverter
from src.util.constants import Directory


class CTSDataloader(BaseModel, BaseDataLoader):

    """
    Data taken from López-Oriona Á, Vilar JA, D’Urso P (2023). “Hard and soft clustering of categorical time series
based on two novel distances with an application to biological sequences.” Information Sciences,
624, 467–492.

    """

    path: str

    class Config:
        arbitrary_types_allowed=True

    @field_validator('path')
    def _set_path(cls, v):

        filepath = str(Directory.DATA / v)

        return filepath

    @pickle_cache(ignore_caching=True)
    def execute(self) -> dict:

        data = pd.read_csv(self.path).drop(columns=['Unnamed: 0'])

        return {'event': data}


if __name__ == '__main__':

    ts_data = CTSDataloader(
        path="ProteinSequences.csv"
    ).execute()

    print(ts_data)
