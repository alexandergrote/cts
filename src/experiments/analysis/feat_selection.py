import pandas as pd
from pydantic import BaseModel

from src.util.custom_logging import console
from src.experiments.analysis.base import BaseAnalyser


class FeatureSelection(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        metric_col, dataset_col, n_features_col = 'metrics.roc_auc_score', 'params.fetch_data.class_name', 'params.preprocess.params.selector.params.n_features'

        all_columns = [metric_col, n_features_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        data_copy_grouped = data_copy.groupby([dataset_col, n_features_col])[metric_col].agg(['mean', 'std']).reset_index()

        console.log(data_copy_grouped)