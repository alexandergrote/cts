import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Optional, Dict, Any, Tuple, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator

from src.experiments.analysis.base import BaseAnalyser
from src.util.constants import Directory

# Set black and white style
sns.set_style('white')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'black'])


class SupportThresholdImpactData(BaseModel):
    min_support: List[float] = Field(description="Minimum support threshold values")
    runtime: List[float] = Field(description="Runtime in seconds for each threshold")
    accuracy: List[float] = Field(description="Classification accuracy for each threshold")
    
    # Field validators (V2 style)
    @field_validator('min_support')
    @classmethod
    def validate_min_support(cls, v: List[float]) -> List[float]:
        if any(x < 0 or x > 1 for x in v):
            raise ValueError("All minimum support values must be between 0 and 1")
        return v
    
    @field_validator('runtime')
    @classmethod
    def validate_runtime(cls, v: List[float]) -> List[float]:
        if any(x <= 0 for x in v):
            raise ValueError("All runtime values must be positive")
        return v
    
    @field_validator('accuracy')
    @classmethod
    def validate_accuracy(cls, v: List[float]) -> List[float]:
        if any(x < 0 or x > 1 for x in v):
            raise ValueError("All accuracy values must be between 0 and 1")
        return v
    
    # Model validator to check that all lists have the same length
    @model_validator(mode='after')
    def validate_lengths(self) -> 'SupportThresholdImpactData':
        lists = [self.min_support, self.runtime, self.accuracy]
        if len(set(len(lst) for lst in lists)) > 1:
            raise ValueError("All lists must have the same length")
        return self
    
    def validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual validation for DataFrame"""
        # Check column existence
        required_cols = ['min_support', 'runtime', 'accuracy']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Value range checks
        if (df['min_support'] < 0).any() or (df['min_support'] > 1).any():
            raise ValueError("All minimum support values must be between 0 and 1")
        
        if (df['runtime'] <= 0).any():
            raise ValueError("All runtime values must be positive")
        
        if (df['accuracy'] < 0).any() or (df['accuracy'] > 1).any():
            raise ValueError("All accuracy values must be between 0 and 1")
        
        return df
    
    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'min_support': self.min_support,
            'runtime': self.runtime,
            'accuracy': self.accuracy
        })
        return self.validate_df(df)


class SupportThresholdImpactPlot(BaseModel):

    data_list: List[SupportThresholdImpactData]
    
    def to_df(self) -> List[pd.DataFrame]:
        """Convert all data instances to DataFrames with validation"""
        return [data.to_df() for data in self.data_list]
    
    def plot_multiple(self, 
                     titles: List[str],
                     save_path: str = "sensitivity_plots.pdf",
                     figsize: Tuple[int, int] = (18, 6), 
                     style: str = "whitegrid",
                     color_runtime: str = "black",
                     color_accuracy: str = "black") -> Dict[str, Any]:
        """
        Plot three support threshold impact plots in a row
        
        Args:
            titles: List of titles for each plot (if None, default titles are used)
            figsize: Figure size (width, height) in inches
            save_path: Path to save the figure, if None, the figure is not saved
            style: Seaborn style to use
            color_runtime: Color for runtime line
            color_accuracy: Color for accuracy line
            
        Returns:
            Dictionary containing figure and axes objects
        """
        # Check if we have enough data
        if len(self.data_list) < 1:
            raise ValueError("At least one dataset must be provided")
        
        # Convert to DataFrames and validate
        dfs = self.to_df()
        
        # Set seaborn style
        sns.set_style(style)
        
        # Create figure and axes
        fig, axes = plt.subplots(1, len(self.data_list), figsize=figsize)

        plt.subplots_adjust(wspace=2)
        
        # Handle the case where there's only one plot
        if len(self.data_list) == 1:
            axes = [axes]
        
        # Plot each dataset
        for i, (df, ax, title) in enumerate(zip(dfs, axes, titles)):
            
            # First axis (runtime)
            ax1 = ax
            ax1.set_xlabel("Minimum Support Threshold")
            ax1.set_ylabel("Runtime (seconds)", color=color_runtime)
            sns.lineplot(x="min_support", y="runtime", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Runtime", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to 0.05 intervals
            ax1.set_xticks(np.arange(0, 1.05, 0.05))
            ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)])
            
            # Second axis (accuracy)
            ax2 = ax1.twinx()
            ax2.set_ylabel("AUC", color=color_accuracy)
            sns.lineplot(x="min_support", y="accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax2, label="Accuracy", 
                         linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color_accuracy)
            
            # Format y-ticks to show 2 decimal places
            #ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            # Title
            ax1.set_title(title)
            
            # Create custom legend

            # Remove default legends created by seaborn
            if ax1.get_legend():
                ax1.get_legend().remove()
            if ax2.get_legend():
                ax2.get_legend().remove()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Store handles and labels for the common legend
            if i == 0:  # Only need to get these once
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                legend_handles = lines1 + lines2
                legend_labels = labels1 + labels2
        
        # Create a common legend for all subplots
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                   ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.98),
                   frameon=True, facecolor='white', edgecolor='black',
                   handlelength=3)
        
        # Tight layout with space for the legend
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")
        

class Sensitivity(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)
        data_copy.reset_index(inplace=True, drop=True)

        dataset_col, n_samples_col = 'dataset', 'n_samples'
        feat_extractor_col = 'feat_extractor'
        feat_selector_col = 'feat_selector'

        scenario_col = 'case'

        metric_duration, metric_memory = 'duration', 'memory'
        metric_col_auc, metric_col_auc_v = 'metrics.roc_auc_score', 'AUC'
        metric_col_f1, metric_col_f1_v = 'metrics.f1_score', 'F1 Score'
        rel_support = 'min_support_rel'
        
        # create mapping for renaming
        mapping = {
            'params.fetch_data.class_name': dataset_col,
            'metrics.feature_selection_duration': metric_duration,
            'metrics.feature_selection_max_memory': metric_memory,
            'params.preprocess.params.extractor.params.prefixspan_config.params.min_support_rel': rel_support,
            metric_col_auc: metric_col_auc_v,
            metric_col_f1: metric_col_f1_v,
        }

        data_copy.rename(columns=mapping, inplace=True)

        datasets = []
        titles = []

        for dataset in data_copy[dataset_col].unique():

            data_copy_sub = data_copy[data_copy[dataset_col] == dataset]

            support_impact_data = SupportThresholdImpactData(
                min_support=data_copy_sub[rel_support],
                runtime=data_copy_sub[metric_duration],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].capitalize()}"
                

            datasets.append(support_impact_data)
            titles.append(title)

        plotter = SupportThresholdImpactPlot(
            data_list=datasets
        )

        plotter.plot_multiple(
            titles=titles
        )

if __name__ == '__main__':

    # Create three datasets
    data1 = SupportThresholdImpactData(
        min_support=[0.1, 0.2, 0.3, 0.4, 0.5],
        runtime=[10.2, 8.5, 6.3, 4.1, 2.8],
        accuracy=[0.82, 0.79, 0.75, 0.72, 0.65]
    )
    
    data2 = SupportThresholdImpactData(
        min_support=[0.1, 0.2, 0.3, 0.4, 0.5],
        runtime=[12.5, 9.8, 7.2, 5.1, 3.0],
        accuracy=[0.88, 0.85, 0.80, 0.74, 0.68]
    )
    
    data3 = SupportThresholdImpactData(
        min_support=[0.1, 0.2, 0.3, 0.4, 0.5],
        runtime=[8.7, 7.2, 5.8, 3.9, 2.3],
        accuracy=[0.79, 0.76, 0.73, 0.69, 0.64]
    )
    
    # Create the plotter with three datasets
    plotter = SupportThresholdImpactPlot(data_list=[data1, data2, data3])
    
    # Plot with custom titles
    plotter.plot_multiple(
        titles=["Dataset A", "Dataset B", "Dataset C"],
        figsize=(18, 6),
        save_path="sensitivity_plots.pdf"
    )
