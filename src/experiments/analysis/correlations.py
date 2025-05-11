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
            sns.regplot(data=data, x='Confidence Delta', y='Average Target Value', color='grey', ax=ax_obj)
            ax_obj.set_ylim(0, 1)
            

        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

        plt.tight_layout(pad=1.5)
        plt.savefig(Directory.FIGURES_DIR / 'scatter_correlation.pdf', dpi=300)
        #plt.show()
        plt.close()

        # Define colors for consistency
        colors = ['blue', 'green', 'purple']

        # Filter only synthetic datasets
        synthetic_records = [(exp_name, data, p_value, corr_value) for exp_name, data, p_value, corr_value in records if "synthetic" in exp_name]
        
        if synthetic_records:
            # Create a grid of subplots based on the number of synthetic datasets
            num_synthetic = len(synthetic_records)
            fig, axes = plt.subplots(3, num_synthetic, figsize=(6*num_synthetic, 15))
            
            for j, y_var in enumerate(['chi_squared', 'entropy', "fisher"]):
                for i, (exp_name, data, _, _) in enumerate(synthetic_records):
                    
                    order = 2
                    lowess = False
                    
                    # Create regression plot
                    sns.regplot(
                        x='Confidence Delta', 
                        y=y_var, 
                        data=data, 
                        scatter_kws={'alpha': 0.5, 'color': colors[j]},
                        line_kws={'color': colors[j]},
                        order=order,
                        lowess=lowess,
                        ax=axes[j, i] if num_synthetic > 1 else axes[j]
                    )
                    
                    title = f"{exp_name.split('_')[-1].capitalize()}_{y_var}"
                    
                    # Set title and labels
                    if num_synthetic > 1:
                        axes[j, i].set_title(title, fontsize=12)
                        axes[j, i].set_xlabel('Confidence Delta', fontsize=10)
                        axes[j, i].set_ylabel(f'{y_var}', fontsize=10)
                    else:
                        axes[j].set_title(title, fontsize=12)
                        axes[j].set_xlabel('Confidence Delta', fontsize=10)
                        axes[j].set_ylabel(f'{y_var}', fontsize=10)

            # Adjust layout
            plt.tight_layout()
            plt.savefig(Directory.FIGURES_DIR / f'metrics_comparison.pdf')
            #plt.show()
