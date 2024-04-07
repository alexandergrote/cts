import pandas as pd
import numpy as np
import optuna

from typing import Union, Type
from pydantic import BaseModel, field_validator, model_validator
from pydantic.v1.utils import deep_update
from sklearn.model_selection import train_test_split

from src.util.dynamic_import import DynamicImport
from src.model.base import BaseProcessModel, BaseHyperParams
from src.evaluation.base import BaseEvaluator


class LSTMHyperParams(BaseHyperParams):

    @staticmethod
    def get_params_for_study(trial: optuna.Trial):

        params = {
            'params': {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
                'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0),
                'model': {'params': {'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256, 512])}}
        }}

        return params
    
class XGBHyperParams(BaseHyperParams):

    @staticmethod
    def get_params_for_study(trial: optuna.Trial):

        params = {
            'params': {
                'params': {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0),
                }
            }
        }

        return params

class HyperTuner(BaseModel, BaseProcessModel):

    n_trials: int 
    timeout: int = 600

    model: Union[dict, Type[BaseProcessModel]]
    hyperparams: Union[dict, Type[BaseHyperParams]]
    evaluator: Union[dict, Type[BaseEvaluator]]

    @field_validator('evaluator', 'hyperparams')
    def _set_model(cls, v):
        return DynamicImport.import_class_from_dict(dictionary=v)
    

    def _objective(self, trial: optuna.Trial, model: dict, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):

        params = self.hyperparams.get_params_for_study(trial)

        # update model with new params
        model = deep_update(model, params)

        # create model
        model: BaseProcessModel = DynamicImport.import_class_from_dict(dictionary=model) 

        model.fit(
            x_train=x_train,
            y_train=y_train
        )

        y_pred_proba = model._predict_proba(x_test)
        y_pred = model._predict(x_test)

        result = self.evaluator.evaluate(
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            y_test=y_test
        )

        return result['metrics']['f1_score']

    def _run_hyperparameter_search(self, sequences: pd.DataFrame, targets: pd.Series) -> optuna.study.Study:

        # split train test
        x_train, x_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )

        x_train = pd.DataFrame(x_train)
        y_train = pd.Series(y_train)
        x_test = pd.DataFrame(x_test)
        y_test = pd.Series(y_test)
        

        study = optuna.create_study(
            direction="maximize",
            study_name="Hyperparameter Tuning",
        )

        study.optimize(
            lambda trial: self._objective(
                trial, 
                self.model,
                x_train,
                x_test,
                y_train,
                y_test
            ),

            n_trials=self.n_trials,
            timeout=self.timeout
        )

        
        return study
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):

        study = self._run_hyperparameter_search(
            sequences=x_train,
            targets=y_train
        )

        params = self.hyperparams.get_params_for_study(study.best_trial)

        # update model with new params
        model = deep_update(self.model, params)

        # create model
        self.model = DynamicImport.import_class_from_dict(dictionary=model)

        self.model.fit(
            x_train=x_train,
            y_train=y_train
        ) 

        return kwargs

    def _predict(self, x_test, **kwargs):
        return self.model._predict(x_test)
    
    def _predict_proba(self, x_test, **kwargs):
        return self.model._predict_proba(x_test)


if __name__ == '__main__':

    import yaml
    from src.fetch_data.synthetic import DataLoader
    from src.util.constants import Directory, replace_placeholder_in_dict

    # get constants
    with open(Directory.CONFIG / 'constants\synthetic.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    # get synthetic data
    with open(Directory.CONFIG / 'fetch_data\synthetic.yaml', 'r') as file:
        config = yaml.safe_load(file)['params']


    for _, value in cfg.items():

        placeholder = value['placeholder']
        replacement = value['value']

        config = replace_placeholder_in_dict(
            dictionary=config,
            placeholder=placeholder,
            replacement=replacement
        )

    data_loader = DataLoader(**config)
    data = data_loader.execute()['data']

    mapping = {event: i+1 for i, event in enumerate(data['event_column'].unique())}
    data['event_column'] = data['event_column'].map(mapping)

    # get sequences from data
    data.sort_values(by='timestamp', inplace=True)
    sequences = data.groupby('id_column')['event_column'].apply(list).to_list()
    targets = data.groupby('id_column')['target'].apply(lambda x: np.unique(x)[0]).to_list()

    # get model config
    with open(Directory.CONFIG / 'model\lstm.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    # get eval config
    with open(Directory.CONFIG / 'evaluation\ml.yaml', 'r') as file:
        eval_config = yaml.safe_load(file)

    tuner = HyperTuner(
        n_trials=1,
        timeout=600,
        model=model_config,
        evaluator=eval_config
    )

    tuner.fit(
        sequences=sequences,
        targets=targets
    )

    tuner.predict(sequences)
    tuner.predict_proba(sequences)
