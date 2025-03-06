from pydantic import BaseModel
from typing import List

from src.experiments.util.types import Experiment


class ExperimentFactory(BaseModel):

    @classmethod
    def create_feature_selection_experiments(cls) -> List[Experiment]:
        
        experiments = []

        for dataset in ['synthetic', 'malware', 'churn']:

            for model in ["logistic_regression", "xgb", "nb"]:

                for selection_method in ["mutinfo", "rf", "mrmr", "self"]:

                    for encoding in ["oh", 'spm']:

                        for n_features in [1,2,3,4,5,6,7,8,9,10]:

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

        for selection_method in ["mutinfo_prefix", "rf_prefix", "mrmr_prefix", "self_spm"]:

            for sample_size in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
            
                exp_name = f'cost__preprocess__{selection_method}__sample_size__{sample_size}'

                overrides = [
                    f'fetch_data=synthetic',
                    f'fetch_data.params.n_samples={sample_size}',
                    f'preprocess={selection_method}',
                    f'preprocess.params.selector.params.n_features=10',
                    f'train_test_split=stratified',
                    f'train_test_split.params.random_state=0',
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

    @classmethod
    def create_correlation_experiments(cls) -> List[Experiment]:

        experiments = []        

        for dataset in ["synthetic", "malware", "churn"]:
            
            exp_name = f'correlation_{dataset}'

            overrides = [
                f'fetch_data={dataset}',
                f'preprocess=self_spm',
                'preprocess.params.extractor.params.prefixspan_config.params.min_support_abs=100',
                'preprocess.params.extractor.params.prefixspan_config.params.min_support_rel=0.05',
                f'train_test_split=stratified',
                f'train_test_split.params.random_state=0',
                f'model=random_chance',
                f'evaluation=ml',
                f'export=mlflow',
                f'export.params.experiment_name={exp_name}'
            ]

            experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments

    @classmethod
    def create_benchmark_experiments(cls) -> List[Experiment]:

        experiments = []

        for dataset in ['synthetic', 'malware', 'churn']:

            for model in ["lstm", "xgb_oh", "xgb_spm"]:

                exp_name = f'benchmark__{dataset}__{model}'

                preprocess = "NA"
                additional_overriedes = []  # Initialize additional_overrides as an empty list

                if model == 'lstm':
                    
                    preprocess = f'baseline_lstm'

                    vocab_size = 2

                    if dataset == 'synthetic':
                        vocab_size = 16
                    elif dataset == 'malware':
                        vocab_size = 260
                    elif dataset == 'churn':
                        vocab_size = 6
                    else:
                        raise ValueError(f"Unknown dataset: {dataset}")  # Raise an error for unknown datasets

                    additional_overriedes = [
                        f'model.params.model.params.model.params.vocab_size={vocab_size}'
                    ]

                if model == 'xgb_oh':
                    preprocess = 'baseline'

                if model == 'xgb_spm':
                    preprocess = 'self_spm'
                
                if model in ['xgb_oh', 'xgb_spm']:
                    model = 'xgb'

                cached_functions = [
                    'src.preprocess.extraction.ts_features.py.SPMFeatureSelector._encode_train',
                    'src.preprocess.extraction.ts_features.py.SPMFeatureSelector._encode_test',
                    'src.fetch_data.synthetic.py.DataLoader.get_data',
                    'src.fetch_data.churn.py.ChurnDataloader.get_data',
                    'src.fetch_data.malware.py.MalwareDataloader.get_data'
                ]

                cached_functions_str = "[" + ','.join(cached_functions) + "]" # without the square brackets, hydra does not recognize it as a list

                overrides = [
                    f'fetch_data={dataset}',
                    f'preprocess={preprocess}',
                    f'train_test_split=stratified',
                    f'train_test_split.params.random_state=0', # ,1,2,3,4
                    f'model={model}_tuned',
                    f'evaluation=ml',
                    f'export=mlflow',
                    f'export.params.experiment_name={exp_name}',
                    f'env.cached_functions={cached_functions_str}'
                ]

                overrides.extend(additional_overriedes)

                experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments