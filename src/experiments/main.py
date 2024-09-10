
import os
import sys

from copy import copy
from pydantic import BaseModel
from typing import Optional, List

from src.util.constants import Directory


class Experiment(BaseModel):

    name: str
    overrides: Optional[List[str]] = None

    @property
    def command(self):

        if self.overrides is None:
            self.overrides = []

        overrides_copy = copy(self.overrides)

        # defaulting to os.system because compose api is limited
        # and does not allow --multirun
        final_command = sys.executable + f" {str(Directory.SRC / 'main.py')} --multirun " + " ".join(overrides_copy)

        return final_command

    def run(self):

        return_code = os.system(self.command)

        if return_code != 0:
            raise RuntimeError(f"Experiment {self.name} failed with return code {return_code}")

    @classmethod
    def create_feature_selection_experiments(cls) -> List["Experiment"]:
        
        experiments = []

        for dataset in ['synthetic', 'malware', 'churn']:

            for model in ["logistic_regression", "xgb", "nb"]:

                for selection_method in ["mutinfo", "rf", "mrmr", "self"]:

                    for encoding in ["oh", 'spm']:

                        for n_features in ["null",1,2,3,4,5,6,7,8,9,10]:

                            exp_name = f'{encoding}__{dataset}__preprocess__{selection_method}__model__{model}__features__{n_features}'

                            overrides = [
                                f'fetch_data={dataset}',
                                f'preprocess={selection_method}_{encoding}',
                                f'preprocess.params.selector.params.n_features={n_features}',
                                f'train_test_split=stratified',
                                f'train_test_split.params.random_state=0,1,2,3,4',
                                f'model={model}',
                                f'evaluation=ml',
                                f'export=mlflow',
                                f'export.params.experiment_name={exp_name}'
                            ]

                            if encoding == "spm":
                                overrides.append(f'preprocess.params.extractor.params.prefixspan_config.params.min_support_abs=100')

                            experiments.append(cls(name=exp_name, overrides=overrides))

        return experiments