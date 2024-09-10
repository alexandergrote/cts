import pandas as pd
from pydantic import BaseModel

from src.util.custom_logging import console
from src.experiments.analysis.base import BaseAnalyser


class FeatureSelection(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        dataset_col, n_features_col = 'params.fetch_data.class_name', 'params.preprocess.params.selector.params.n_features'
        model_col = 'params.model.params.model'
        feat_extractor_col = 'params.preprocess.params.extractor.class_name'
        feat_selector_col = 'params.preprocess.params.selector.class_name'

        metric_col_auc, metric_col_f1 = 'metrics.roc_auc_score', 'metrics.f1_score'
        metric_col = [metric_col_auc, metric_col_f1]

        # format result
        data_copy[dataset_col] = data_copy[dataset_col].apply(lambda x: x.split('.')[-1])
        data_copy[model_col] = data_copy[model_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_extractor_col] = data_copy[feat_extractor_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_selector_col] = data_copy[feat_selector_col].apply(lambda x: x.split('.')[-1])
        data_copy[n_features_col] = data_copy[n_features_col].replace({'None': -1}).astype(int)

        df_to_plot = data_copy.groupby([dataset_col, model_col, feat_extractor_col, feat_selector_col, n_features_col])[metric_col].agg(['mean', 'std']).reset_index()
        df_to_plot.sort_values(by=[dataset_col, model_col, feat_extractor_col, feat_selector_col, n_features_col], inplace=True)
        df_to_plot.columns = ['dataset','model', 'feat_extractor', 'feat_selector', 'n_features', 'metric_col_auc_mean', 'metric_col_auc_std', 'metric_col_f1_mean', 'metric_col_f1_std']
        
        print(
            df_to_plot.to_markdown()
        )