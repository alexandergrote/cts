import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import BaseModel

from src.experiments.analysis.base import BaseAnalyser


class CostBenefit(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        dataset_col, n_samples_col = 'params.fetch_data.class_name', 'params.fetch_data.params.n_samples'
        feat_extractor_col = 'params.preprocess.params.extractor.class_name'
        feat_selector_col = 'params.preprocess.params.selector.class_name'

        metric_duration, metric_memory = 'metrics.feature_selection_duration', 'metrics.feature_selection_max_memory'
        metric_col = [metric_duration, metric_memory]

        # format result
        data_copy[dataset_col] = data_copy[dataset_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_extractor_col] = data_copy[feat_extractor_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_selector_col] = data_copy[feat_selector_col].apply(lambda x: x.split('.')[-1])
        data_copy[n_samples_col] = data_copy[n_samples_col].replace({'None': -1}).astype(int)

        df_to_plot = data_copy.groupby([dataset_col, feat_extractor_col, feat_selector_col, n_samples_col])[metric_col].agg(['mean', 'std']).reset_index()
        df_to_plot.sort_values(by=[dataset_col, feat_extractor_col, feat_selector_col, n_samples_col], inplace=True)
        df_to_plot.columns = ['dataset', 'feat_extractor', 'feat_selector', 'n_samples', 'duration_mean', 'duration_std', 'memory_mean', 'memory_std']
        
        print(
            df_to_plot.to_markdown()
        )

        # Create a figure with 2 columns
        fig, axs = plt.subplots(nrows=len(df_to_plot.dataset.unique()), ncols=2, figsize=(12, 4*len(df_to_plot.dataset.unique())))
        axs = axs.reshape(1, -1)

        plt.subplots_adjust(hspace=0.5)

        # Iterate over each dataset
        for i, dataset in enumerate(df_to_plot.dataset.unique()):
            
            dataset_df = df_to_plot[df_to_plot.dataset == dataset]
            
            # Time based plot
            sns.lineplot(data=dataset_df, x='n_samples', y='duration_mean', hue='feat_extractor', ax=axs[i, 0])
            axs[i, 0].set_title(f'{dataset} - Duration')
            
            # Memory based plot
            sns.lineplot(data=dataset_df, x='n_samples', y='memory_mean', hue='feat_extractor', ax=axs[i, 1])
            axs[i, 1].set_title(f'{dataset} - Memory')

        plt.show()
