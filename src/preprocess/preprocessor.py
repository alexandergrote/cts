from pydantic import BaseModel, field_validator
from typing import Union, Dict, Type, Any
from typing_extensions import Annotated

from src.preprocess.base import BaseFeatureSelector, BaseFeatureEncoder
from src.util.dynamic_import import DynamicImport
from src.util.logging import console


class FeatureMaker(BaseModel):

    extractor: Union[Dict[str, dict], Any]
    selector: Union[Dict[str, dict],  Any]

    class Config:
        arbitrary_types_allowed=True

    @field_validator("extractor", "selector")
    def _init_model(cls, v):
        return DynamicImport.import_class_from_dict(dictionary=v)

    def execute(self, **kwargs):

        console.log("Extracting features")
        output = self.extractor._encode(**kwargs)

        console.log("Selecting features")
        result = self.selector._select_features(**output)

        output['data'] = result

        return output


