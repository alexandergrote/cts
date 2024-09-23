import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from pydantic import BaseModel

from src.experiments.analysis.base import BaseAnalyser


sns.set_style('white')


class CostBenefit(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)
        data_copy.reset_index(inplace=True, drop=True)

        dataset_col, n_samples_col = 'dataset', 'n_samples'
        feat_extractor_col = 'feat_extractor'
        feat_selector_col = 'feat_selector'

        metric_duration, metric_memory = 'duration', 'memory'
        metric_col = [metric_duration, metric_memory]

        # create mapping for renaming
        mapping = {
            'params.fetch_data.class_name': dataset_col,
            'params.preprocess.params.extractor.class_name': feat_extractor_col,
            'params.preprocess.params.selector.class_name': feat_selector_col,
            'params.fetch_data.params.n_samples': n_samples_col,
            'metrics.feature_selection_duration': metric_duration,
            'metrics.feature_selection_max_memory': metric_memory
        }

        data_copy.rename(columns=mapping, inplace=True)

        # format result
        data_copy[dataset_col] = data_copy[dataset_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_extractor_col] = data_copy[feat_extractor_col].apply(lambda x: x.split('.')[-1])
        data_copy[feat_selector_col] = data_copy[feat_selector_col].apply(lambda x: x.split('.')[-1])
        data_copy[n_samples_col] = data_copy[n_samples_col].replace({'None': -1}).astype(int)

        df_to_plot = data_copy.groupby([dataset_col, feat_extractor_col, feat_selector_col, n_samples_col])[metric_col].agg(['mean', 'std']).reset_index()
        df_to_plot.sort_values(by=[dataset_col, feat_extractor_col, feat_selector_col, n_samples_col], inplace=True)

        # rename multi-level columns for improved readability
        df_to_plot.columns = ['_'.join(col).strip() for col in df_to_plot.columns.values]
        
        print(
            df_to_plot.to_markdown()
        )

        # Create a figure with 2 columns
        _, axs = plt.subplots(nrows=len(data_copy[dataset_col].unique()), ncols=2, figsize=(12, 4*len(data_copy[dataset_col].unique())))
        axs = axs.reshape(1, -1)

        plt.subplots_adjust(hspace=0.5)

        # Iterate over each dataset
        for i, dataset in enumerate(data_copy[dataset_col].unique()):

            dataset_df = data_copy[data_copy[dataset_col] == dataset]

            # Adjust the number of grey tones based on the unique values in your hue column        
            grey_palette = sns.color_palette(['grey'] * data_copy[feat_extractor_col].nunique())  
            
            # Define custom dash patterns
            dash_pattern_list = [(2, 2), (4, 2), (6, 2), (8, 2), (10, 2)]

            # map dash patterns to hue values
            num_unique_hues = data_copy[feat_extractor_col].nunique()

            if num_unique_hues > len(dash_pattern_list):
                raise ValueError(f'Number of unique hues ({num_unique_hues}) is greater than the number of dash patterns ({len(dash_pattern_list)})')

            titles = ['Duration', 'Memory']

            for j, (metric_col, title) in enumerate(zip(metric_col, titles)):

                # Create the plot without specifying dashes
                plot = sns.lineplot(data=dataset_df, x=n_samples_col, y=metric_col, hue=feat_extractor_col, ax=axs[i, j], errorbar='sd', palette=grey_palette)

                # Set dash patterns manually
                for idx, line in enumerate(plot.get_lines()):
                    line.set_dashes(dash_pattern_list[idx % len(dash_pattern_list)])

                legend_handles, _ = axs[i, j].get_legend_handles_labels()
                for idx, handle in enumerate(legend_handles):
                    handle.set_dashes(dash_pattern_list[idx % len(dash_pattern_list)])
                axs[i, j].legend(handles=legend_handles)

                axs[i, j].set_title(title)

        plt.show()
