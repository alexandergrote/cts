import pandas as pd
import numpy as np

from pydantic import BaseModel, field_validator
from typing import Union, Dict, Any
from scipy.stats import mannwhitneyu

from src.preprocess.extraction.ts_features import SPMFeatureSelector
from src.util.dynamic_import import DynamicImport
from src.util.custom_logging import console
from src.util.datasets import DatasetSchema
from src.util.profile import max_memory_tracker, time_tracker, Tracker
from src.preprocess.util.datasets import DatasetRulesSchema


def get_pvalue_mwu_test(x: pd.Series) -> float:

    obs = np.abs(np.array(x))
    values = values = np.zeros_like(obs)

    test = mannwhitneyu(obs, values, alternative='greater')

    return test.pvalue


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

        if not isinstance(self.extractor, SPMFeatureSelector):

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

        else:

            console.log("Extracting features")
            output = self.extractor.execute(**kwargs)

            console.log("Selecting features")
            output = self.selector.execute(**output)

            # add output to kwargs
            kwargs = {**kwargs, **output}


        kwargs['x_train'], kwargs['y_train'] = self._get_x_y(output['data_train'])
        kwargs['x_test'], kwargs['y_test'] = self._get_x_y(output['data_test'])

        # save correlation data, needed for paper analysis
        if kwargs['case_name'].startswith('correlation_'):

            if not isinstance(self.extractor, SPMFeatureSelector):
                raise ValueError("Feature maker must be SPMFeatureSelector")

            rules = kwargs['rules'].data.copy(deep=True)
            pvalues = kwargs['pvalues'].data.copy(deep=True)
            bootstrap_rounds = kwargs['bootstrap_rounds']


            rules['id_column'] = rules['id_column'].apply(lambda x: '_'.join(x))
            pvalues['id_column'] = pvalues['id_column'].apply(lambda x: '_'.join(x))
            pvalues.columns = ['delta_conf_' + el for el in pvalues.columns]

            rules = rules.merge(pvalues, left_on='id_column', right_on='delta_conf_id_column')

            rules['avg_delta_confidence'] = rules[DatasetRulesSchema.delta_confidence].apply(lambda x: sum(x)/len(x))
            rules['avg_chi_squared'] = rules[DatasetRulesSchema.chi_squared].apply(lambda x: sum(x)/len(x))
            rules['avg_chi_squared_p'] = rules[DatasetRulesSchema.chi_squared_p_values].apply(lambda x: sum(x)/len(x))
            rules['avg_entropy'] = rules[DatasetRulesSchema.entropy].apply(lambda x: sum(x)/len(x))
            rules['avg_fisher'] = rules[DatasetRulesSchema.fisher_odds_ratio].apply(lambda x: sum(x)/len(x))
            rules['avg_fisher_p'] = rules[DatasetRulesSchema.fisher_odds_ratio_p_values].apply(lambda x: sum(x)/len(x))
            rules['avg_phi'] = rules['phi'].apply(lambda x: sum(x)/len(x))
            rules['avg_leverage'] = rules[DatasetRulesSchema.leverage].apply(lambda x: sum(x)/len(x))

            rules['phi_p_values'] = rules['phi'].apply(lambda x: get_pvalue_mwu_test(x))
            rules['leverage_p_values'] = rules[DatasetRulesSchema.leverage].apply(lambda x: get_pvalue_mwu_test(x))

            rules['avg_support'] = rules.support.apply(lambda x: sum(x)/len(x))
            

            delta_conf_mapping = dict(zip(rules['id_column'], rules['avg_delta_confidence']))
            delta_conf_p_mapping = dict(zip(rules['id_column'], rules['delta_conf_p_values']))
            chi_quared_mapping = dict(zip(rules['id_column'], rules['avg_chi_squared']))
            chi_squared_p_mapping = dict(zip(rules['id_column'], rules['avg_chi_squared_p']))
            entropy_mapping = dict(zip(rules['id_column'], rules['avg_entropy']))
            fisher_mapping = dict(zip(rules['id_column'], rules['avg_fisher']))
            fisher_p_mapping = dict(zip(rules['id_column'], rules['avg_fisher_p']))
            phi_mapping = dict(zip(rules['id_column'], rules['avg_phi']))
            phi_p_mapping = dict(zip(rules['id_column'], rules['phi_p_values']))
            leverage_mapping = dict(zip(rules['id_column'], rules['avg_leverage']))
            leverage_p_mapping = dict(zip(rules['id_column'], rules['leverage_p_values']))

            y_train = kwargs['y_train'].copy(deep=True)
            x_train = kwargs['x_train'].copy(deep=True)

            records = []

            for col in x_train.columns:

                avg_target = y_train[x_train[col]].mean()
                delta_conf = delta_conf_mapping[col]
                delta_conf_p = delta_conf_p_mapping[col]
                chi_squared = chi_quared_mapping[col]
                chi_squared_p = chi_squared_p_mapping[col]
                entropy = entropy_mapping[col]
                phi = phi_mapping[col]
                phi_p = phi_p_mapping[col]
                fisher = fisher_mapping[col]
                fisher_p = fisher_p_mapping[col]
                leverage = leverage_mapping[col]
                leverage_p = leverage_p_mapping[col]

                records.append({
                    'pattern': col,
                    'avg_target': avg_target,
                    'delta_conf': delta_conf,
                    'delta_conf_p': delta_conf_p,
                    'chi_squared': chi_squared,
                    'chi_squared_p': chi_squared_p,
                    'entropy': entropy,
                    'fisher': fisher,
                    'fisher_p': fisher_p,
                    'phi': phi,
                    'phi_p': phi_p,
                    'leverage': leverage,
                    'leverage_p': leverage_p
                })

            data = pd.DataFrame(records)
            data.to_csv("correlations.csv", index=False)

            # top leverage analysis

            leverage = rules.sort_values(by='leverage', ascending=False).leverage.iloc[0]
            pattern = rules.sort_values(by='leverage', ascending=False).id_column.iloc[0]

            for i, b_round in enumerate(bootstrap_rounds):
                print('-'*10, 'Bootstrap round', i, '-'*10, 'Pattern:', pattern)
                print(b_round.n_samples, b_round.n_samples_neg, b_round.n_samples_pos)
                for freq_pattern in b_round.freq_patterns:
                    pattern_name = '_'.join(freq_pattern.antecedent + freq_pattern.consequent)
                    if pattern_name == pattern:
                        print(freq_pattern.support, freq_pattern.support_pos, freq_pattern.support_neg)
                        print(freq_pattern.support / b_round.n_samples, freq_pattern.support_pos / b_round.n_samples, freq_pattern.support_neg / b_round.n_samples)
                        print(freq_pattern.support_pos / b_round.n_samples - b_round.n_samples_pos / b_round.n_samples *  freq_pattern.support / b_round.n_samples)
                        break

        return kwargs


