import pandas as pd

from pydantic import BaseModel, field_validator
from typing import Union, Dict, Any

from src.preprocess.extraction.ts_features import SPMFeatureSelector
from src.util.dynamic_import import DynamicImport
from src.util.custom_logging import console
from src.util.datasets import DatasetSchema
from src.util.profile import max_memory_tracker, time_tracker, Tracker


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

        tracker = Tracker()

        with time_tracker(tracker=tracker), max_memory_tracker(tracker=tracker):

            console.log("Extracting features")
            output = self.extractor.execute(**kwargs)

            console.log("Selecting features")
            output = self.selector.execute(**output)

        # add output to kwargs
        kwargs = {**kwargs, **output}

        # add tracker metrics to kwargs, needed for paper analysis
        kwargs['feature_selection_duration'] = tracker.time_taken_seconds
        kwargs['feature_selection_max_memory'] = tracker.max_memory_mb

        kwargs['x_train'], kwargs['y_train'] = self._get_x_y(output['data_train'])
        kwargs['x_test'], kwargs['y_test'] = self._get_x_y(output['data_test'])

        # save correlation data, needed for paper analysis
        if kwargs['case_name'].startswith('correlation_'):

            if not isinstance(self.extractor, SPMFeatureSelector):
                raise ValueError("Feature maker must be SPMFeatureSelector")

            rules = kwargs['rules'].data.copy(deep=True)
            rules['id_column'] = rules['id_column'].apply(lambda x: '_'.join(x))
            rules['avg_delta_confidence'] = rules['delta_confidence'].apply(lambda x: sum(x)/len(x))
            rules['avg_chi_squared'] = rules['chi_squared'].apply(lambda x: sum(x)/len(x))
            rules['avg_entropy'] = rules['entropy'].apply(lambda x: sum(x)/len(x))
            rules['avg_fisher'] = rules['fisher_odds_ratio'].apply(lambda x: sum(x)/len(x))
            rules['avg_phi'] = rules['phi'].apply(lambda x: sum(x)/len(x))
            
            

            delta_conf_mapping = dict(zip(rules['id_column'], rules['avg_delta_confidence']))
            chi_quared_mapping = dict(zip(rules['id_column'], rules['avg_chi_squared']))
            entropy_mapping = dict(zip(rules['id_column'], rules['avg_entropy']))
            fisher_mapping = dict(zip(rules['id_column'], rules['avg_fisher']))
            phi_mapping = dict(zip(rules['id_column'], rules['avg_phi']))

            y_train = kwargs['y_train'].copy(deep=True)
            x_train = kwargs['x_train'].copy(deep=True)

            records = []

            for col in x_train.columns:

                avg_target = y_train[x_train[col]].mean()
                delta_conf = delta_conf_mapping[col]
                chi_squared = chi_quared_mapping[col]
                entropy = entropy_mapping[col]
                phi = phi_mapping[col]

                records.append({
                    'pattern': col,
                    'avg_target': avg_target,
                    'delta_conf': delta_conf,
                    'chi_squared': chi_squared,
                    'entropy': entropy,
                    'fisher': fisher_mapping[col],
                    'phi': phi,
                })

            data = pd.DataFrame(records)
            data.to_csv("correlations.csv", index=False)

        return kwargs


