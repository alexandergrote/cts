import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import BaseModel
from scipy.stats import pearsonr

from src.experiments.analysis.base import BaseAnalyser
from src.util.constants import Directory
from src.util.mlflow_util import uri_to_path


class Correlations(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)
        data_copy.reset_index(inplace=True, drop=True)

        # track data 
        records = []

        # run info meta.artifact_uri
        for _, row in data_copy.iterrows():

            exp_name = row['params.export.params.experiment_name']
            artifact_uri = row['meta.artifact_uri']  

            directory = uri_to_path(artifact_uri)
            data = pd.read_csv(directory / "correlations.csv")
            
            pearsonr_result = pearsonr(data['avg_target'], data['delta_conf'], alternative='greater')
            p_value = pearsonr_result.pvalue
            corr_value = pearsonr_result.correlation

            print(exp_name, 'corr_value:', corr_value,'p-value:', p_value)

            # add to records
            records.append((exp_name, data, p_value, corr_value))

        sns.set(font_scale=2.5)
        sns.set_style('white')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'black'])

        f, axes = plt.subplots(1, len(records), sharey=False, figsize=(24,6))
        
        # reorder records to match other experiments
        records = records[2:] + records[:2]

        for idx, (exp_name, data, p_value, corr_value) in enumerate(records):

            data.rename({'avg_target': 'Average Target Value', 'delta_conf': 'Confidence Delta', 'mean_ranking': 'Ranking'}, axis=1, inplace=True)
            #sns.scatterplot(data=data[mask], x='Confidence Delta', y='Deviation From Average Target', color='grey', linewidth=1, edgecolor='grey', markers='x')

            p_value_verbose = f"p<{p_value:.2f}"

            if p_value < 0.01:
                p_value_verbose = "p<0.01"

            if p_value < 0.001:
                p_value_verbose = "p<0.001"

            corr_verbose = round(corr_value, 2)           
    

            title = f"{exp_name.split('_')[-1].capitalize()}\n(ρ={corr_verbose}, {p_value_verbose})"
            

            ax_obj = axes if len(records) == 1 else axes[idx]

            ax_obj.title.set_text(title)
            sns.regplot(data=data, x='Confidence Delta', y='Average Target Value', color='grey', ax=ax_obj, scatter_kws={'rasterized': True})
            ax_obj.set_ylim(0, 1)
            

        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

        plt.tight_layout(pad=1.5)
        plt.savefig(Directory.FIGURES_DIR / 'scatter_correlation.pdf', dpi=300)
        #plt.show()
        plt.close()

        # Define colors for consistency in black and white
        colors = ['grey', 'grey', 'grey', 'grey']

        # Filter only synthetic datasets
        malware_records = [(exp_name, data, p_value, corr_value) for exp_name, data, p_value, corr_value in records if "malwa" in exp_name]
        
        if malware_records:
            # Create a single row of subplots for all metrics
            metrics = ['chi_squared', 'entropy', "fisher", "phi"]
            num_plots = len(metrics)
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
            
            # Use only the first synthetic dataset
            exp_name, data, _, _ = malware_records[0]

            sns.set(font_scale=2.0)
            sns.set_style('white')

            for j, y_var in enumerate(metrics):
                order = 2
                lowess = False

                if y_var == 'phi':
                    order = 1

                if y_var == 'fisher':
                    order = 1#3
                    lowess = True

                # Create regression plot with different line styles for black and white
                line_styles = ['-', '-', '-', '-']
                marker_styles = ['o', 'o', 'o', 'o']

                color = sns.color_palette(['grey'])

                sns.regplot(
                    x='Confidence Delta', 
                    y=y_var, 
                    data=data, 
                    scatter_kws={'color': color, 'marker': marker_styles[j], 'rasterized': True},
                    line_kws={'color': 'black', 'linestyle': line_styles[j], 'linewidth': 2},
                    order=order,
                    lowess=lowess,
                    ax=axes[j],
                )

                y_var_mapping = {
                    'chi_squared': 'Chi-Squared',
                    'entropy': 'Entropy',
                    'fisher': 'Fisher Odds Ratio',
                    'phi': 'Phi Coefficient',
                }

                y_var = y_var_mapping.get(y_var, y_var)
                
                title = f"{y_var}"
                
                # Set title and labels with increased font size
                axes[j].set_title(title)
                axes[j].set_xlabel('Confidence Delta')
                axes[j].set_ylabel(f'{y_var}')
                axes[j].tick_params(axis='both', which='major')


                if y_var == 'Entropy':
                    axes[j].set_ylim(0, 1)
                

            # Adjust layout
            plt.tight_layout()
            plt.savefig(Directory.FIGURES_DIR / f'metrics_comparison.pdf')
            #plt.show()
