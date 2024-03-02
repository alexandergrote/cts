import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Union
from typing_extensions import Literal

from src.preprocess.base import BasePreprocessor
from src.util.dynamic_import import DynamicImport
from src.util.filepath_converter import FilepathConverter
from src.util.constants import Directory
from src.util.logging import Pickler


class MergeDatasetConfig(BaseModel):

    merge_on: Union[str, List[str]]  
    ignore_columns: Optional[List[str]]
    keep_columns: Optional[List[str]]

    @model_validator(mode='after')
    def _validate_columns(self):

        ignore_columns = self.ignore_columns
        keep_columns = self.keep_columns

        # check if only one option is specified
        # leaving both None is also valid
        if ignore_columns is not None and keep_columns is not None:
            raise ValueError("Only one of ignore_columns or keep_columns can be specified")

        return self

    def get_columns(self):

        columns = self.merge_on
        if isinstance(self.merge_on, str):
            columns = [self.merge_on]

        if self.keep_columns is not None:
            columns += self.keep_columns

        if self.ignore_columns is not None:
            columns += self.ignore_columns

        # get only unique values
        columns = list(set(columns))

        return columns


class MergeDataset(BaseModel):

    data: pd.DataFrame
    config: MergeDatasetConfig

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _validate_columns(data: pd.DataFrame, config: MergeDatasetConfig):

        columns = config.get_columns()

        # check if columns exist in dataframe
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column {column} not in dataframe")

    @model_validator(mode='after')
    def _set_data(self):

        data = self.data
        config = self.config

        MergeDataset._validate_columns(data, config)

        if config.ignore_columns is not None:
            data.drop(config.ignore_columns, axis=1, inplace=True)

        if config.keep_columns is not None:
            data = data[config.keep_columns]

        self.data = data

        return self


class Merger(BaseModel, BasePreprocessor):

    dataset_left_config: Union[dict, MergeDatasetConfig]
    dataset_right_config: Union[dict, MergeDatasetConfig]

    how: Literal['left', 'right'] = 'left'

    n_rows: Optional[int]

    filename_merge: str
    merge_params: Optional[dict]  

    override: bool = True
    save: bool = False 

    @field_validator("filename_merge")
    def _check_filepath_requirements(cls, v):

        converter = FilepathConverter(target_dir=Directory.DATA)
        
        return converter.get_filepath(v)

    @field_validator("merge_params")
    def _set_none_to_dict(cls, v):

        if v is None:
            return {}
        return v

    @field_validator("dataset_left_config", "dataset_right_config")
    def _init_classes(cls, v):
        return DynamicImport.import_class_from_dict(v)

    def execute(self, *, event: pd.DataFrame, case: pd.DataFrame, **kwargs) -> dict:

        if not self.override:
            return {'data': pd.read_parquet(path=self.filename_merge)}

        dataset_left = MergeDataset(data=event, config=self.dataset_left_config)
        dataset_right = MergeDataset(data=case, config=self.dataset_right_config)

        data = dataset_left.data.merge(
            dataset_right.data,
            left_on=dataset_left.config.merge_on,
            right_on=dataset_right.config.merge_on,
            how=self.how
        )

        """Paper Analysis Start"""

        rules = kwargs['rules']

        columns = [col for col in data.columns if '-->' in col]

        result = {}

        for column in columns:
            result[column] = data[data[column]]['target'].sum() / sum(data['target'])

        result = pd.Series(result)
        result.name = 'avg_target'

        result = rules.merge(result, left_on='index', right_index=True)

        Pickler.write(result, "rules_conf_target.pickle")

        """Paper Analysis End"""

        if self.n_rows is not None:
            data = data.head(self.n_rows)

        if self.save:
            data.to_parquet(self.filename_merge)

        return {'data': data}
