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


class MultiTestingImpactData(BaseModel):
    multitesting: List[bool] = Field(description="Boolean indication of whether multitesting was applied or not")
    accuracy: List[float] = Field(description="Classification accuracy for each threshold")
    number_of_features: List[int] = Field(description="Number of features selected for each threshold")

    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'multitesting': self.multitesting,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy
        })
        
        # Berechne relative Änderungen
        # Verwende den Mittelwert aller "multitesting == False" als Referenzwert
        no_correction_mask = df['multitesting'] == False
        
        if no_correction_mask.any():
            # Referenzwerte (Mittelwert aller "multitesting == False")
            ref_features = df.loc[no_correction_mask, 'number_of_features'].mean()
            ref_accuracy = df.loc[no_correction_mask, 'accuracy'].mean()
        else:
            # Fallback, wenn keine "multitesting == False" vorhanden sind
            ref_features = df['number_of_features'].iloc[0]
            ref_accuracy = df['accuracy'].iloc[0]
        
        # Berechne relative Änderungen in Prozent
        df['rel_number_of_features'] = (df['number_of_features'] / ref_features - 1) * 100
        df['rel_accuracy'] = (df['accuracy'] / ref_accuracy - 1) * 100
        
        return df


class MultiTestingImpactPlot(BaseModel):

    data_list: List[MultiTestingImpactData]
    
    def to_df(self) -> List[pd.DataFrame]:
        """Convert all data instances to DataFrames with validation"""
        return [data.to_df() for data in self.data_list]
    
    def plot_multiple(self, 
                     titles: List[str],
                     save_path: str = "sensitivity_multitesting_plots.pdf",
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
            
            # First axis (number of features - relative change)
            ax1 = ax
            ax1.set_xlabel("Multitesting Correction")
            ax1.set_ylabel("Change in Number of Features (%)", color=color_runtime)
            
            # Convert boolean to categorical labels and ensure consistent order
            df['multitesting_label'] = df['multitesting'].apply(lambda x: "With Correction" if x else "No Correction")
            
            # Sortiere die Kategorien, damit "No Correction" immer links und "With Correction" immer rechts ist
            df['multitesting_label'] = pd.Categorical(df['multitesting_label'], 
                                                     categories=["No Correction", "With Correction"], 
                                                     ordered=True)
            
            # Use lineplot with relative values
            sns.lineplot(x="multitesting_label", y="rel_number_of_features", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Number of Features", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Horizontale Linie bei 0% (keine Änderung)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Adjust y-axis to show relative changes better
            y_min = min(df['rel_number_of_features'].min() * 1.1, -5)  # Mindestens -5% anzeigen
            y_max = max(df['rel_number_of_features'].max() * 1.1, 5)   # Mindestens +5% anzeigen
            ax1.set_ylim(y_min, y_max)
            
            # Second axis (accuracy - relative change)
            ax2 = ax1.twinx()
            ax2.set_ylabel("Change in AUC (%)", color=color_accuracy)
            
            # Add lines connecting the points for better visualization
            sns.lineplot(x="multitesting_label", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax2, label="AUC", 
                         linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color_accuracy)
            
            # Format y-ticks to show 1 decimal place
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            # Horizontale Linie bei 0% (keine Änderung)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Get the range of relative accuracy values
            y_min, y_max = df['rel_accuracy'].min(), df['rel_accuracy'].max()
            # Add a larger buffer to ensure all points are visible and separated
            buffer = 2  # 2% buffer
            y_min = min(y_min - buffer, -1)  # Mindestens -1% anzeigen
            y_max = max(y_max + buffer, 1)   # Mindestens +1% anzeigen
            
            # Set y-axis limits
            ax2.set_ylim(y_min, y_max)
            
            # Get unique accuracy values and create custom ticks
            import matplotlib.ticker as ticker
            
            # Use fewer ticks to avoid duplicates
            ax2.yaxis.set_major_locator(ticker.LinearLocator(4))
            
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
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")
        

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
        
        # Calculate relative changes
        # Use the data point with the smallest min_support as reference
        min_idx = df['min_support'].idxmin()
        ref_runtime = df.loc[min_idx, 'runtime']
        ref_accuracy = df.loc[min_idx, 'accuracy']
        
        # Calculate relative changes in percent
        df['rel_runtime'] = (df['runtime'] / ref_runtime - 1) * 100
        df['rel_accuracy'] = (df['accuracy'] / ref_accuracy - 1) * 100
        
        return self.validate_df(df)


class SupportThresholdImpactPlot(BaseModel):

    data_list: List[SupportThresholdImpactData]
    
    def to_df(self) -> List[pd.DataFrame]:
        """Convert all data instances to DataFrames with validation"""
        return [data.to_df() for data in self.data_list]
    
    def plot_multiple(self, 
                     titles: List[str],
                     save_path: str = "sensitivity_support_plots.pdf",
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
            
            # First axis (runtime - relative change)
            ax1 = ax
            ax1.set_xlabel("Minimum Support Threshold")
            ax1.set_ylabel("Change in Runtime (%)", color=color_runtime)
            sns.lineplot(x="min_support", y="rel_runtime", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Runtime", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to 0.05 intervals
            ax1.set_xticks(np.arange(0, 1.05, 0.05))
            ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)])
            
            # Horizontal line at 0% (no change)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Adjust y-axis to show relative changes better
            y_min = min(df['rel_runtime'].min() * 1.1, -5)  # At least -5% display
            y_max = max(df['rel_runtime'].max() * 1.1, 5)   # At least +5% display
            ax1.set_ylim(y_min, y_max)
            
            # Second axis (accuracy - relative change)
            ax2 = ax1.twinx()
            ax2.set_ylabel("Change in AUC (%)", color=color_accuracy)
            sns.lineplot(x="min_support", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax2, label="AUC", 
                         linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color_accuracy)
            
            # Format y-ticks to show 1 decimal place
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            # Horizontal line at 0% (no change)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Get the range of relative accuracy values
            y_min, y_max = df['rel_accuracy'].min(), df['rel_accuracy'].max()
            # Add a larger buffer to ensure all points are visible and separated
            buffer = 2  # 2% buffer
            y_min = min(y_min - buffer, -1)  # At least -1% display
            y_max = max(y_max + buffer, 1)   # At least +1% display
            
            # Set y-axis limits
            ax2.set_ylim(y_min, y_max)
            
            # Get unique accuracy values and create custom ticks
            import matplotlib.ticker as ticker
            
            # Use fewer ticks to avoid duplicates
            ax2.yaxis.set_major_locator(ticker.LinearLocator(4))
            
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
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")


class BufferImpactData(BaseModel):
    buffer: List[float] = Field(description="Criterion buffer values")
    accuracy: List[float] = Field(description="Classification accuracy for each threshold")
    number_of_features: List[int] = Field(description="Number of features selected for each threshold")

    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'buffer': self.buffer,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy
        })
        return df


class BootstrapRoundsData(BaseModel):
    bootstrap_rounds: List[int] = Field(description="Number of bootstrap rounds")
    accuracy: List[float] = Field(description="Classification accuracy for each number of rounds")
    number_of_features: List[int] = Field(description="Number of features selected for each number of rounds")

    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'bootstrap_rounds': self.bootstrap_rounds,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy
        })
        return df
        

class BufferImpactPlot(BaseModel):

    data_list: List[BufferImpactData]
    
    def to_df(self) -> List[pd.DataFrame]:
        """Convert all data instances to DataFrames with validation"""
        return [data.to_df() for data in self.data_list]
    
    def plot_multiple(self, 
                     titles: List[str],
                     save_path: str = "sensitivity_buffer_plots.pdf",
                     figsize: Tuple[int, int] = (18, 6), 
                     style: str = "whitegrid",
                     color_runtime: str = "black",
                     color_accuracy: str = "black") -> Dict[str, Any]:
        """
        Plot buffer thresholds impact plots in a row
        
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
            ax1.set_xlabel("Buffer Threshold")
            ax1.set_ylabel("Number of Features", color=color_runtime)
            sns.lineplot(x="buffer", y="number_of_features", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Number of Features", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to 0.05 intervals
            ax1.set_xticks(np.arange(0, 1.05, 0.05))
            ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)])
            
            # Set y1-ticks to integer values only
            from matplotlib.ticker import MaxNLocator
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Second axis (accuracy)
            ax2 = ax1.twinx()
            ax2.set_ylabel("AUC", color=color_accuracy)
            sns.lineplot(x="buffer", y="accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax2, label="AUC", 
                         linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color_accuracy)
            
            # Format y-ticks to show 2 decimal places and only unique values
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            # Get the range of accuracy values
            y_min, y_max = df['accuracy'].min(), df['accuracy'].max()
            # Add a small buffer to ensure all points are visible
            buffer = 0.02
            y_min = max(0, y_min - buffer)
            y_max = min(1, y_max + buffer)
            
            # Set y-axis limits
            ax2.set_ylim(y_min, y_max)
            
            # Get unique accuracy values and create custom ticks
            import matplotlib.ticker as ticker
            
            # Use fewer ticks to avoid duplicates
            ax2.yaxis.set_major_locator(ticker.LinearLocator(4))
            
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
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")


class BootstrapRoundsPlot(BaseModel):

    data_list: List[BootstrapRoundsData]
    
    def to_df(self) -> List[pd.DataFrame]:
        """Convert all data instances to DataFrames with validation"""
        return [data.to_df() for data in self.data_list]
    
    def plot_multiple(self, 
                     titles: List[str],
                     save_path: str = "sensitivity_bootstrap_plots.pdf",
                     figsize: Tuple[int, int] = (18, 6), 
                     style: str = "whitegrid",
                     color_runtime: str = "black",
                     color_accuracy: str = "black") -> Dict[str, Any]:
        """
        Plot bootstrap rounds impact plots in a row
        
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
            
            # First axis (number of features)
            ax1 = ax
            ax1.set_xlabel("Bootstrap Rounds")
            ax1.set_ylabel("Number of Features", color=color_runtime)
            sns.lineplot(x="bootstrap_rounds", y="number_of_features", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Number of Features", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to specific values: 1, 5, 10, 15, 20
            ax1.set_xticks([1, 5, 10, 15, 20])
            
            # Set y1-ticks to integer values only
            from matplotlib.ticker import MaxNLocator
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Second axis (accuracy)
            ax2 = ax1.twinx()
            ax2.set_ylabel("AUC", color=color_accuracy)
            sns.lineplot(x="bootstrap_rounds", y="accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax2, label="AUC", 
                         linestyle="--", linewidth=2)
            ax2.tick_params(axis="y", labelcolor=color_accuracy)
            
            # Format y-ticks to show 2 decimal places and only unique values
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            # Get the range of accuracy values
            y_min, y_max = df['accuracy'].min(), df['accuracy'].max()
            # Add a small buffer to ensure all points are visible
            buffer = 0.02
            y_min = max(0, y_min - buffer)
            y_max = min(1, y_max + buffer)
            
            # Set y-axis limits
            ax2.set_ylim(y_min, y_max)
            
            # Get unique accuracy values and create custom ticks
            import matplotlib.ticker as ticker
            
            # Use fewer ticks to avoid duplicates
            ax2.yaxis.set_major_locator(ticker.LinearLocator(4))
            
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
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
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
        n_features_selected = 'metrics.n_features_selected'
        criterion_buffer = 'criterion_buffer'
        
        # create mapping for renaming
        mapping = {
            'params.fetch_data.class_name': dataset_col,
            'metrics.feature_selection_duration': metric_duration,
            'metrics.feature_selection_max_memory': metric_memory,
            'params.preprocess.params.extractor.params.prefixspan_config.params.min_support_rel': rel_support,
            'params.preprocess.params.extractor.params.criterion_buffer': 'criterion_buffer',
            metric_col_auc: metric_col_auc_v,
            metric_col_f1: metric_col_f1_v,
            n_features_selected: 'N Features Selected',
        }

        data_copy.rename(columns=mapping, inplace=True)

        ## min support params
        exp_names = data_copy['params.export.params.experiment_name']
        mask = exp_names.str.contains('min_support', na=False)
        data_copy_support = data_copy[mask]

        datasets = []
        titles = []

        for dataset in data_copy_support[dataset_col].unique():

            data_copy_sub = data_copy_support[data_copy_support[dataset_col] == dataset]

            support_impact_data = SupportThresholdImpactData(
                min_support=data_copy_sub[rel_support],
                runtime=data_copy_sub[metric_duration],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            datasets.append(support_impact_data)
            titles.append(title)

        if len(datasets) > 0:

            plotter = SupportThresholdImpactPlot(
                data_list=datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('multitest', na=False)
        data_copy_multitest = data_copy[mask]

        datasets = []
        titles = []

        for dataset in data_copy_multitest[dataset_col].unique():

            data_copy_sub = data_copy_multitest[data_copy_multitest[dataset_col] == dataset]

            # Keine Sortierung mehr nötig, da wir den Mittelwert aller "multitesting == False" verwenden
            
            multitesting_data = MultiTestingImpactData(
                multitesting=data_copy_sub['params.export.params.experiment_name'].str.contains('True', na=False),
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            datasets.append(multitesting_data)
            titles.append(title)

        if len(datasets) > 0:

            plotter = MultiTestingImpactPlot(
                data_list=datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('buffer', na=False)
        data_copy_buffer = data_copy[mask]

        datasets = []
        titles = []

        for dataset in data_copy_buffer[dataset_col].unique():

            data_copy_sub = data_copy_buffer[data_copy_buffer[dataset_col] == dataset]

            buffer_data = BufferImpactData(
                buffer=data_copy_sub['criterion_buffer'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            datasets.append(buffer_data)
            titles.append(title)

        if len(datasets) > 0:

            plotter = BufferImpactPlot(
                data_list=datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('bootstrap', na=False)
        data_copy_bootstrap = data_copy[mask]

        datasets = []
        titles = []

        for dataset in data_copy_bootstrap[dataset_col].unique():

            data_copy_sub = data_copy_bootstrap[data_copy_bootstrap[dataset_col] == dataset]

            bootstrap_data = BootstrapRoundsData(
                bootstrap_rounds=data_copy_sub['params.preprocess.params.extractor.params.bootstrap_repetitions'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                
            datasets.append(bootstrap_data)
            titles.append(title)

        if len(datasets) > 0:

            plotter = BootstrapRoundsPlot(
                data_list=datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

if __name__ == '__main__':

    # Create three datasets for SupportThresholdImpactPlot
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
        save_path="sensitivity_plots_tmp.pdf"
    )

    # Create a dataset for BufferImpactPlot
    buffer_data = BufferImpactData(
        buffer=[0.05, 0.1, 0.15, 0.2],
        accuracy=[0.78, 0.82, 0.84, 0.86],
        number_of_features=[10, 15, 20, 25]
    )
    
    # Create the plotter with one dataset
    buffer_plotter = BufferImpactPlot(data_list=[buffer_data])
    
    # Plot with custom title
    buffer_plotter.plot_multiple(
        titles=["Buffer Impact"],
        figsize=(18, 6),
        save_path="buffer_impact_plot_tmp.pdf"
    )
    
    # Create a dataset for BootstrapRoundsPlot
    bootstrap_data = BootstrapRoundsData(
        bootstrap_rounds=[1, 5, 10, 15, 20],
        accuracy=[0.75, 0.80, 0.83, 0.85, 0.86],
        number_of_features=[8, 12, 15, 18, 20]
    )
    
    # Create the plotter with one dataset
    bootstrap_plotter = BootstrapRoundsPlot(data_list=[bootstrap_data])
    
    # Plot with custom title
    bootstrap_plotter.plot_multiple(
        titles=["Bootstrap Rounds Impact"],
        figsize=(18, 6),
        save_path="bootstrap_rounds_plot_tmp.pdf"
    )
