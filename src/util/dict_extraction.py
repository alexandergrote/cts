from pydantic import BaseModel
from src.util.constants import YamlField
from typing import Tuple


class DictExtraction(BaseModel):
    @staticmethod
    def get_class_obj_and_params(dictionary: dict) -> Tuple[str, dict]:

        class_obj = dictionary[YamlField.CLASS_NAME.value]
        class_params = dictionary.get(YamlField.PARAMS.value)

        if class_params is None:
            return class_obj, {}

        return class_obj, class_params