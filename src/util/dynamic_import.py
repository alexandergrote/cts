from pydantic import BaseModel
from typing import Optional
from src.util.dict_extraction import DictExtraction 


class DynamicImport(BaseModel):
    @staticmethod
    def _get_class_obj(name: str):
        # index of last dot
        dot_index = name.rfind(".")
        class_name = name[dot_index + 1 :]

        mod = __import__(name[:dot_index], fromlist=[class_name])
        class_object = getattr(mod, class_name)

        return class_object

    @staticmethod
    def init_class(name: str, params: Optional[dict]):

        class_object = DynamicImport._get_class_obj(name=name)

        if params is None:
            return class_object()

        return class_object(**params)
    
    @staticmethod
    def import_class_from_dict(dictionary: dict):

        class_obj, class_params = DictExtraction.get_class_obj_and_params(dictionary=dictionary)

        return DynamicImport.init_class(name=class_obj, params=class_params)
