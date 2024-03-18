from pydantic import BaseModel, field_validator
from typing import Union, Dict, Any

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
        output = self.extractor.execute(**kwargs)

        console.log("Selecting features")
        output = self.selector.execute(**output)

        return output


