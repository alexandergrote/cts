from pydantic import BaseModel
from typing import List, Union, Dict

from src.util.dynamic_import import DynamicImport
from src.util.constants import YamlField
from src.util.logging import console


class Preprocessor(BaseModel):

    steps: Union[List[dict], Dict[str, dict]]

    def execute(self, **kwargs):

        # define result variable
        output = {}

        # get iterable
        steps = self.steps
        if isinstance(self.steps, dict):
            steps = [v for _, v in self.steps.items()]

        for step in steps:
            console.log(f"Executing {step[YamlField.CLASS_NAME.value]}")

            preprocessor = DynamicImport.import_class_from_dict(step)
            output = preprocessor.execute(**kwargs)
            kwargs.update(output)

        return output


