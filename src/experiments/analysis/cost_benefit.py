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

        scenario_col = 'case'

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

        # add scenario column
        data_copy[scenario_col] = data_copy[feat_selector_col]

        # Create a figure with 2 columns
        fig, axs = plt.subplots(nrows=len(data_copy[dataset_col].unique()), ncols=2, figsize=(12, 4*len(data_copy[dataset_col].unique())))
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
            num_unique_hues = data_copy[scenario_col].nunique()

            if num_unique_hues > len(dash_pattern_list):
                raise ValueError(f'Number of unique hues ({num_unique_hues}) is greater than the number of dash patterns ({len(dash_pattern_list)})')

            titles = ['Duration', 'Peak Memory']

            for j, (metric_col, title) in enumerate(zip(metric_col, titles)):
                
                xlabel = 'Number of Samples'

                dataset_df[scenario_col] = dataset_df[scenario_col].str.replace("FeatSelection", "")

                dataset_df[scenario_col] = dataset_df[scenario_col].replace({
                    'MutInfo': 'Mutual Information',
                    'RF': 'Random Forest',
                    'MRMR': 'mRMR',
                    'TimeSeriesFeatureSelection': 'Delta Confidence'
                })
                
                if j == 0:
                    # Set marker styles manually
                    marker_styles = ['D', 's', 'o', 'X']
                    data2plot = dataset_df
                    ylabel = 'Duration (s)'

                else:
                    marker_styles = ['D', 'X']
                    mask = dataset_df[[feat_extractor_col, n_samples_col]].drop_duplicates().index
                    data2plot = dataset_df.loc[mask]
                    ylabel = 'Peak Memory (MB)'
                    data2plot[scenario_col] = data2plot[scenario_col].replace({
                        'Mutual Information': 'Prefix Based Methods'
                    })
                
                # Create the plot without specifying dashes
                plot = sns.lineplot(data=data2plot, x=n_samples_col, y=metric_col, hue=scenario_col, ax=axs[i, j], errorbar='sd', palette=grey_palette)

                # Set dash patterns manually
                #for idx, line in enumerate(plot.get_lines()):
                #    line.set_dashes(dash_pattern_list[idx % len(dash_pattern_list)])

                for idx, line in enumerate(plot.get_lines()):
                    line.set_marker(marker_styles[idx % len(marker_styles)])


                axs[i, j].set_title(title)
                axs[i, j].legend().set_visible(True)
                axs[i, j].set_xlabel(xlabel)
                axs[i, j].set_ylabel(ylabel)

        # Create a legend and center it above the plots
        #legend_handles, _ = axs[0, 0].get_legend_handles_labels()
        #fig.legend(legend_handles, data_copy[scenario_col].unique(), loc='upper center', ncol=len(data_copy[scenario_col].unique()))

        plt.show()