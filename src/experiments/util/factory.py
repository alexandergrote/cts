
import sys

from copy import copy
from pydantic import BaseModel
from typing import Optional, List

from src.util.constants import Directory
from src.experiments.util.types import Experiment


class ExperimentFactory(BaseModel):

    @classmethod
    def create_feature_selection_experiments(cls) -> List[Experiment]:
        
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

                            # only the ts feature selection method has this field
                            if encoding == "spm":
                                overrides.append(f'preprocess.params.extractor.params.prefixspan_config.params.min_support_abs=100')

                            experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments
    
    @classmethod
    def create_cost_benefit_experiments(cls) -> List[Experiment]:

        experiments = []        

        for selection_method in ["rf_prefix", "self_spm"]:# ["mutinfo_prefix", "rf_prefix", "mrmr_prefix", "self_spm"]:

            #for sample_size in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
            for sample_size in [1000, 2000, 3000, 4000, 5000]:

                exp_name = f'cost__preprocess__{selection_method}__sample_size__{sample_size}'

                overrides = [
                    f'fetch_data=synthetic',
                    f'fetch_data.params.n_samples={sample_size}',
                    f'preprocess={selection_method}',
                    f'preprocess.params.selector.params.n_features=10',
                    f'train_test_split=stratified',
                    f'train_test_split.params.random_state=0,1,2,3,4',
                    f'model=random_chance',
                    f'evaluation=ml',
                    f'export=mlflow',
                    f'export.params.experiment_name={exp_name}'
                ]

                if selection_method == "self_spm":
                    overrides.append(f'preprocess.params.extractor.params.prefixspan_config.params.min_support_abs=100')
                else:
                    overrides.append(f'preprocess.params.extractor.params.min_support_abs=100')

                experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments
