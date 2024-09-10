import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Union, Dict, Any

from src.util.dynamic_import import DynamicImport
from src.util.custom_logging import console
from src.util.datasets import DatasetSchema


class FeatureMaker(BaseModel):

    extractor: Union[Dict[str, dict], Any]
    selector: Union[Dict[str, dict],  Any]

    class Config:
        arbitrary_types_allowed=True

    @field_validator("extractor", "selector")
    def _init_model(cls, v):
        return DynamicImport.import_class_from_dict(dictionary=v)
    
    def _get_x_y(self, df: pd.DataFrame):
        return df.drop(columns=DatasetSchema.class_column), df[DatasetSchema.class_column]


    def execute(self, **kwargs):

        console.log("Extracting features")
        output = self.extractor.execute(**kwargs)

        console.log("Selecting features")
        output = self.selector.execute(**output)

        kwargs['x_train'], kwargs['y_train'] = self._get_x_y(output['data_train'])
        kwargs['x_test'], kwargs['y_test'] = self._get_x_y(output['data_test'])

        return kwargs


