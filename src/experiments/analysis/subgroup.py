import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import BaseModel

from src.experiments.analysis.base import BaseAnalyser
from src.util.constants import Directory

sns.set_style('white')


class SubGroupBenchmark(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        data_copy = data.copy(deep=True)

        dataset_col, dataset_col_v = 'params.fetch_data.class_name', 'Data'
        n_features_col, n_features_col_v = 'params.preprocess.params.selector.params.n_features', 'Number of Features'
        model_col, model_col_v = 'params.model.params.model.params.model', 'Model'
        feat_extractor_col, feat_extractor_col_v = 'params.preprocess.params.extractor.class_name', 'Feature Extractor'
        feat_selector_col, feat_selector_col_v = 'params.preprocess.params.selector.class_name', 'Feature Selector'

        metric_col_auc, metric_col_auc_v = 'metrics.roc_auc_score', 'AUC'
        metric_col_f1, metric_col_f1_v = 'metrics.f1_score', 'F1 Score'
        metric_col = [metric_col_auc, metric_col_f1]

        # exclude none results
        data_copy.reset_index(inplace=True, drop=True)

        
        # format result
        data_copy[dataset_col] = data_copy[dataset_col].apply(lambda x: x.split('.')[-1])
        data_copy[model_col] = data_copy[model_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_extractor_col] = data_copy[feat_extractor_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_selector_col] = data_copy[feat_selector_col].apply(lambda x: x.split('.')[-1])
        data_copy[n_features_col] = data_copy[n_features_col].replace({'None': -1}).astype(int)

        df_to_plot = data_copy.groupby([dataset_col, model_col, feat_extractor_col, feat_selector_col, n_features_col])[metric_col].agg(['mean', 'std']).reset_index()
        df_to_plot.sort_values(by=[dataset_col, model_col, feat_extractor_col, feat_selector_col, n_features_col], inplace=True)
        df_to_plot.columns = [dataset_col_v, model_col_v, feat_extractor_col_v, feat_selector_col_v, n_features_col_v, metric_col_auc_v, f'{metric_col_auc_v} std', metric_col_f1_v, f'{metric_col_f1_v} std']
        
        print(
            df_to_plot.to_markdown()
        )