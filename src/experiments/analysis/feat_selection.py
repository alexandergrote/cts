import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pydantic import BaseModel

from src.util.constants import Directory
from src.experiments.analysis.base import BaseAnalyser


class FeatureSelection(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        dataset_col, dataset_col_v = 'params.fetch_data.class_name', 'Data'
        n_features_col, n_features_col_v = 'params.preprocess.params.selector.params.n_features', 'Number of Features'
        model_col, model_col_v = 'params.model.params.model', 'Model'
        feat_extractor_col, feat_extractor_col_v = 'params.preprocess.params.extractor.class_name', 'Feature Extractor'
        feat_selector_col, feat_selector_col_v = 'params.preprocess.params.selector.class_name', 'Feature Selector'

        metric_col_auc, metric_col_auc_v = 'metrics.roc_auc_score', 'ROC AUC Score'
        metric_col_f1, metric_col_f1_v = 'metrics.f1_score', 'F1 Score'
        metric_col = [metric_col_auc, metric_col_f1]

        # exclude none results
        mask = data_copy[n_features_col] != 'None'
        data_copy = data_copy[mask]
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

        df_to_plot.replace({

            'MRMRFeatSelection': 'mRMR',
            'MutInfoFeatSelection': 'Mutual Information',
            'RFFeatSelection': 'Random Forest',
            'SPMFeatureSelector': 'Delta Confidence',
            'TimeSeriesFeatureSelection': 'Delta Confidence',
            'ChurnDataloader': 'Churn',
            'MalwareDataloader': 'Malware',
            'DataLoader': 'Synthetic',
            'sklearn.linear_model.LogisticRegression': 'LogisticRegression',
            'sklearn.naive_bayes.GaussianNB': 'Naïve Bayes',
            'xgboost.XGBClassifier': 'XGB',
            
        }, inplace=True)

        grey_palette = sns.color_palette(['grey'] * df_to_plot[feat_selector_col_v].nunique())  # Adjust the number of grey tones based on the unique values in your hue column

        # Define markers for different categories in your hue column
        markers = {
            'mRMR': 'o', 
            'Mutual Information': 'X',
            'Random Forest': 's',  
            'Delta Confidence': 'D' 
        }

        g = sns.FacetGrid(df_to_plot, row=model_col_v, col=dataset_col_v, sharey=False, height=4, sharex=True, despine=False)

        g.map_dataframe(
            sns.lineplot, 
            x=n_features_col_v, 
            y=metric_col_f1_v, 
            hue=feat_selector_col_v, 
            palette=grey_palette, 
            style=feat_selector_col_v,
            markers=markers,
            dashes=False
        )

        # Here we're using only '{col_name}' to display just the value of the column variable
        g.set_titles("{row_name} | {col_name}")

        # Customizing tick labels
        for ax in g.axes.flat:
            ax.set_xticks([i + 1 for i in range(10)])                    # Set x-tick positions
            ax.set_xticklabels([str(i + 1) for i in range(10)])  # Custom x-tick labels
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        handles, labels = g.axes.flat[0].get_legend_handles_labels()

        # Remove any duplicate labels/handles
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):

            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        # Draw the unique legend
        g.fig.legend(handles=unique_handles, loc='upper center', labels=unique_labels, bbox_to_anchor=(0.5, 1.05), ncol=len(unique_labels))

        # Optionally, adjust the figure to make room for the legend if needed
        g.fig.subplots_adjust(top=0.9)

        plt.tight_layout()

        for extension in ['png', 'pdf']:
            plt.savefig(Directory.FIGURES_DIR / f'feature_selection.{extension}', dpi=300, bbox_inches='tight', pad_inches=0.25)
        
        #plt.show()