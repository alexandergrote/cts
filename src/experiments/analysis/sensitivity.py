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
        # Use the mean of all data points with the smallest min_support as reference
        min_support_value = df['min_support'].min()
        min_support_mask = df['min_support'] == min_support_value
        
        # Calculate reference values (mean of all points with minimum support)
        ref_runtime = df.loc[min_support_mask, 'runtime'].mean()
        ref_accuracy = df.loc[min_support_mask, 'accuracy'].mean()
        
        # Calculate relative changes in percent
        df['rel_runtime'] = (df['runtime'] / ref_runtime - 1) * 100
        df['rel_accuracy'] = (df['accuracy'] / ref_accuracy - 1) * 100
        
        return self.validate_df(df)


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
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest buffer value as reference
        min_buffer_value = df['buffer'].min()
        min_buffer_mask = df['buffer'] == min_buffer_value
        
        # Calculate reference values (mean of all points with minimum buffer)
        ref_features = df.loc[min_buffer_mask, 'number_of_features'].mean()
        ref_accuracy = df.loc[min_buffer_mask, 'accuracy'].mean()
        
        # Calculate relative changes in percent
        df['rel_number_of_features'] = (df['number_of_features'] / ref_features - 1) * 100
        df['rel_accuracy'] = (df['accuracy'] / ref_accuracy - 1) * 100
        
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
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest bootstrap_rounds as reference
        min_rounds_value = df['bootstrap_rounds'].min()
        min_rounds_mask = df['bootstrap_rounds'] == min_rounds_value
        
        # Calculate reference values (mean of all points with minimum bootstrap rounds)
        ref_features = df.loc[min_rounds_mask, 'number_of_features'].mean()
        ref_accuracy = df.loc[min_rounds_mask, 'accuracy'].mean()
        
        # Calculate relative changes in percent
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
            
            # Add accuracy line to the same axis (similar to BufferImpactPlot)
            sns.lineplot(x="multitesting_label", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax1, label="AUC", 
                         linestyle="--", linewidth=2)
            
            # Title
            ax1.set_title(title)
            
            # Create custom legend
            # Remove default legends created by seaborn
            if ax1.get_legend():
                ax1.get_legend().remove()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Store handles and labels for the common legend
            if i == 0:  # Only need to get these once
                lines1, labels1 = ax1.get_legend_handles_labels()
                legend_handles = lines1
                legend_labels = labels1
        
        # Create a common legend for all subplots
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                   ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.98),
                   frameon=True, facecolor='white', edgecolor='black',
                   handlelength=3)
        
        # Tight layout with space for the legend
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")
        


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
            
            # Add accuracy line to the same axis (similar to BufferImpactPlot)
            sns.lineplot(x="min_support", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax1, label="AUC", 
                         linestyle="--", linewidth=2)
            
            # Title
            ax1.set_title(title)
            
            # Create custom legend
            # Remove default legends created by seaborn
            if ax1.get_legend():
                ax1.get_legend().remove()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Store handles and labels for the common legend
            if i == 0:  # Only need to get these once
                lines1, labels1 = ax1.get_legend_handles_labels()
                legend_handles = lines1
                legend_labels = labels1
        
        # Create a common legend for all subplots
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                   ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.98),
                   frameon=True, facecolor='white', edgecolor='black',
                   handlelength=3)
        
        # Tight layout with space for the legend
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")


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
            
            # First axis (number of features - relative change)
            ax1 = ax
            ax1.set_xlabel("Buffer Threshold")
            ax1.set_ylabel("Change in Number of Features (%)", color=color_runtime)
            sns.lineplot(x="buffer", y="rel_number_of_features", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Number of Features", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to 0.05 intervals
            ax1.set_xticks(np.arange(0, 1.05, 0.05))
            ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)])
            
            # Horizontal line at 0% (no change)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Adjust y-axis to show relative changes better
            y_min = min(df['rel_number_of_features'].min() * 1.1, -5)  # At least -5% display
            y_max = max(df['rel_number_of_features'].max() * 1.1, 5)   # At least +5% display
            ax1.set_ylim(y_min, y_max)
            
            # Second axis (accuracy - relative change)
            sns.lineplot(x="buffer", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax1, label="AUC", 
                         linestyle="--", linewidth=2)
            
            
            # Get the range of relative accuracy values
            y_min, y_max = df['rel_accuracy'].min(), df['rel_accuracy'].max()
            # Add a larger buffer to ensure all points are visible and separated
            buffer = 2  # 2% buffer
            y_min = min(y_min - buffer, -1)  # At least -1% display
            y_max = max(y_max + buffer, 1)   # At least +1% display
            
            
            # Get unique accuracy values and create custom ticks
            import matplotlib.ticker as ticker
            
            # Title
            ax1.set_title(title)
            
            # Create custom legend

            # Remove default legends created by seaborn
            if ax1.get_legend():
                ax1.get_legend().remove()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Store handles and labels for the common legend
            if i == 0:  # Only need to get these once
                lines1, labels1 = ax1.get_legend_handles_labels()
                legend_handles = lines1
                legend_labels = labels1
        
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
            
            # First axis (number of features - relative change)
            ax1 = ax
            ax1.set_xlabel("Bootstrap Rounds")
            ax1.set_ylabel("Change in Number of Features (%)", color=color_runtime)
            sns.lineplot(x="bootstrap_rounds", y="rel_number_of_features", data=df, marker="o", 
                         color=color_runtime, ax=ax1, label="Number of Features", 
                         linestyle="-", linewidth=2)
            ax1.tick_params(axis="y", labelcolor=color_runtime)
            
            # Set x-ticks to specific values: 1, 5, 10, 15, 20
            ax1.set_xticks([1, 5, 10, 15, 20])
            
            # Horizontal line at 0% (no change)
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Adjust y-axis to show relative changes better
            y_min = min(df['rel_number_of_features'].min() * 1.1, -5)  # At least -5% display
            y_max = max(df['rel_number_of_features'].max() * 1.1, 5)   # At least +5% display
            ax1.set_ylim(y_min, y_max)
            
            # Add accuracy line to the same axis (similar to BufferImpactPlot)
            sns.lineplot(x="bootstrap_rounds", y="rel_accuracy", data=df, marker="s", 
                         color=color_accuracy, ax=ax1, label="AUC", 
                         linestyle="--", linewidth=2)
            
            # Title
            ax1.set_title(title)
            
            # Create custom legend
            # Remove default legends created by seaborn
            if ax1.get_legend():
                ax1.get_legend().remove()
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            # Store handles and labels for the common legend
            if i == 0:  # Only need to get these once
                lines1, labels1 = ax1.get_legend_handles_labels()
                legend_handles = lines1
                legend_labels = labels1
        
        # Create a common legend for all subplots
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                   ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.98),
                   frameon=True, facecolor='white', edgecolor='black',
                   handlelength=3)
        
        # Tight layout with space for the legend
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight")
        

class AllInOnePlot(BaseModel):

    support_data_list: List[SupportThresholdImpactData]
    multitesting_data_list: List[MultiTestingImpactData]
    buffer_data_list: List[BufferImpactData]
    bootstrap_data_list: List[BootstrapRoundsData]

    def plot(self, 
             titles: Optional[List[str]] = None,
             save_path_runtime: str = "sensitivity_runtime_plot.pdf",
             save_path_features: str = "sensitivity_features_plot.pdf",
             figsize_runtime: Tuple[int, int] = (18, 6),  # Single row for runtime
             figsize_features: Tuple[int, int] = (18, 18),  # Three rows for features
             style: str = "whitegrid",
             color_runtime: str = "black",
             color_accuracy: str = "black") -> None:
        """
        Create two combined plots:
        1. Runtime plot (Support Threshold)
        2. Features plot (Multitesting, Buffer, Bootstrap)
        
        Args:
            titles: List of titles for each column (dataset). If None, default titles are used.
            save_path_runtime: Path to save the runtime figure
            save_path_features: Path to save the features figure
            figsize_runtime: Figure size for runtime plot (width, height) in inches
            figsize_features: Figure size for features plot (width, height) in inches
            style: Seaborn style to use
            color_runtime: Color for runtime/number of features line
            color_accuracy: Color for accuracy line
        """
        # Check if we have data for at least one plot type
        if (len(self.support_data_list) == 0 and 
            len(self.multitesting_data_list) == 0 and 
            len(self.buffer_data_list) == 0 and 
            len(self.bootstrap_data_list) == 0):
            raise ValueError("At least one dataset must be provided for at least one plot type")
        
        # Determine the number of columns (datasets) for each plot type
        n_cols_runtime = len(self.support_data_list)
        n_cols_features = max(
            len(self.multitesting_data_list),
            len(self.buffer_data_list),
            len(self.bootstrap_data_list)
        )
        
        # Generate default titles if not provided
        if titles is None:
            titles = [f"Dataset {i+1}" for i in range(max(n_cols_runtime, n_cols_features))]
        elif len(titles) < max(n_cols_runtime, n_cols_features):
            titles.extend([f"Dataset {i+1}" for i in range(len(titles), max(n_cols_runtime, n_cols_features))])
        
        # Set seaborn style
        sns.set_style(style)
        
        # PLOT 1: Runtime Plot (Support Threshold)
        if n_cols_runtime > 0:
            # Create figure with 1 row and n_cols columns
            fig_runtime, axes_runtime = plt.subplots(1, n_cols_runtime, figsize=figsize_runtime)
            
            # Handle the case where there's only one plot
            if n_cols_runtime == 1:
                axes_runtime = [axes_runtime]
            
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.3)
            
            # Support Threshold Impact Plot
            support_dfs = [data.to_df() for data in self.support_data_list]
            
            for i, (df, ax) in enumerate(zip(support_dfs, axes_runtime)):
                # First axis (runtime - relative change)
                ax1 = ax
                ax1.set_xlabel("Minimum Support Threshold")
                ax1.set_ylabel("Change (%)", color=color_runtime)
                sns.lineplot(x="min_support", y="rel_runtime", data=df, marker="o", 
                             color=color_runtime, ax=ax1, label="Runtime", 
                             linestyle="-", linewidth=2)
                
                # Set x-ticks to 0.05 intervals
                ax1.set_xticks(np.arange(0, 1.05, 0.05))
                ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)], rotation=45)
                
                # Horizontal line at 0% (no change)
                ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Add accuracy line to the same axis
                sns.lineplot(x="min_support", y="rel_accuracy", data=df, marker="s", 
                             color=color_accuracy, ax=ax1, label="AUC", 
                             linestyle="--", linewidth=2)
                
                # Title
                ax1.set_title(titles[i])
                
                # Remove default legends created by seaborn
                if ax1.get_legend():
                    ax1.get_legend().remove()
                
                # Add grid for better readability
                ax1.grid(True, alpha=0.3)
            
            # Create a common legend for runtime plot
            lines, labels = axes_runtime[0].get_legend_handles_labels()
            fig_runtime.legend(lines, labels, loc='upper center', 
                       ncol=len(labels), bbox_to_anchor=(0.5, 0.98),
                       frameon=True, facecolor='white', edgecolor='black',
                       handlelength=3)
            
            # Tight layout with space for the legend
            fig_runtime.tight_layout(rect=[0, 0, 1, 0.90])
            
            # Save runtime plot
            plt.figure(fig_runtime.number)
            plt.savefig(Directory.FIGURES_DIR / save_path_runtime, dpi=300, bbox_inches="tight")
        
        # PLOT 2: Features Plot (Multitesting, Buffer, Bootstrap)
        if n_cols_features > 0:
            # Count how many feature plot types we have
            n_rows_features = sum([
                1 if len(self.multitesting_data_list) > 0 else 0,
                1 if len(self.buffer_data_list) > 0 else 0,
                1 if len(self.bootstrap_data_list) > 0 else 0
            ])
            
            if n_rows_features == 0:
                return  # No feature plots to create
            
            # Create figure with n_rows rows and n_cols columns
            fig_features, axes_features = plt.subplots(n_rows_features, n_cols_features, 
                                                      figsize=figsize_features)
            
            # Handle the case where there's only one row
            if n_rows_features == 1:
                if n_cols_features == 1:
                    axes_features = np.array([[axes_features]])
                else:
                    axes_features = np.array([axes_features])
            
            # Adjust spacing between subplots
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            # Track current row
            current_row = 0
            
            # Row: Multitesting Impact Plot
            if len(self.multitesting_data_list) > 0:
                multitesting_dfs = [data.to_df() for data in self.multitesting_data_list]
                
                for i, df in enumerate(multitesting_dfs):
                    if i >= n_cols_features:
                        break
                        
                    ax = axes_features[current_row, i]
                    
                    # First axis (number of features - relative change)
                    ax1 = ax
                    ax1.set_xlabel("Multitesting Correction")
                    ax1.set_ylabel("Change (%)", color=color_runtime)
                    
                    # Convert boolean to categorical labels and ensure consistent order
                    df['multitesting_label'] = df['multitesting'].apply(lambda x: "With Correction" if x else "No Correction")
                    
                    # Sort categories to ensure "No Correction" is always left and "With Correction" is always right
                    df['multitesting_label'] = pd.Categorical(df['multitesting_label'], 
                                                             categories=["No Correction", "With Correction"], 
                                                             ordered=True)
                    
                    # Use lineplot with relative values
                    sns.lineplot(x="multitesting_label", y="rel_number_of_features", data=df, marker="o", 
                                 color=color_runtime, ax=ax1, label="Number of Features", 
                                 linestyle="-", linewidth=2)
                    
                    # Horizontal line at 0% (no change)
                    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Add accuracy line to the same axis
                    sns.lineplot(x="multitesting_label", y="rel_accuracy", data=df, marker="s", 
                                 color=color_accuracy, ax=ax1, label="AUC", 
                                 linestyle="--", linewidth=2)
                    
                    # Title
                    ax1.set_title(titles[i])
                    
                    # Remove default legends created by seaborn
                    if ax1.get_legend():
                        ax1.get_legend().remove()
                    
                    # Add grid for better readability
                    ax1.grid(True, alpha=0.3)
                
                # Hide unused axes in this row
                for i in range(len(multitesting_dfs), n_cols_features):
                    axes_features[current_row, i].axis('off')
                
                # Add row title
                fig_features.text(0.02, 0.9 - (current_row * 0.3), "Multitesting", fontsize=14, fontweight='bold')
                
                current_row += 1
            
            # Row: Buffer Impact Plot
            if len(self.buffer_data_list) > 0:
                buffer_dfs = [data.to_df() for data in self.buffer_data_list]
                
                for i, df in enumerate(buffer_dfs):
                    if i >= n_cols_features:
                        break
                        
                    ax = axes_features[current_row, i]
                    
                    # First axis (number of features - relative change)
                    ax1 = ax
                    ax1.set_xlabel("Buffer Threshold")
                    ax1.set_ylabel("Change (%)", color=color_runtime)
                    sns.lineplot(x="buffer", y="rel_number_of_features", data=df, marker="o", 
                                 color=color_runtime, ax=ax1, label="Number of Features", 
                                 linestyle="-", linewidth=2)
                    
                    # Set x-ticks to 0.05 intervals
                    ax1.set_xticks(np.arange(0, 1.05, 0.05))
                    ax1.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 1.05, 0.05)], rotation=45)
                    
                    # Horizontal line at 0% (no change)
                    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Add accuracy line to the same axis
                    sns.lineplot(x="buffer", y="rel_accuracy", data=df, marker="s", 
                                 color=color_accuracy, ax=ax1, label="AUC", 
                                 linestyle="--", linewidth=2)
                    
                    # Title
                    ax1.set_title(titles[i])
                    
                    # Remove default legends created by seaborn
                    if ax1.get_legend():
                        ax1.get_legend().remove()
                    
                    # Add grid for better readability
                    ax1.grid(True, alpha=0.3)
                
                # Hide unused axes in this row
                for i in range(len(buffer_dfs), n_cols_features):
                    axes_features[current_row, i].axis('off')
                
                # Add row title
                fig_features.text(0.02, 0.9 - (current_row * 0.3), "Buffer Threshold", fontsize=14, fontweight='bold')
                
                current_row += 1
            
            # Row: Bootstrap Rounds Plot
            if len(self.bootstrap_data_list) > 0:
                bootstrap_dfs = [data.to_df() for data in self.bootstrap_data_list]
                
                for i, df in enumerate(bootstrap_dfs):
                    if i >= n_cols_features:
                        break
                        
                    ax = axes_features[current_row, i]
                    
                    # First axis (number of features - relative change)
                    ax1 = ax
                    ax1.set_xlabel("Bootstrap Rounds")
                    ax1.set_ylabel("Change (%)", color=color_runtime)
                    sns.lineplot(x="bootstrap_rounds", y="rel_number_of_features", data=df, marker="o", 
                                 color=color_runtime, ax=ax1, label="Number of Features", 
                                 linestyle="-", linewidth=2)
                    
                    # Set x-ticks to specific values: 1, 5, 10, 15, 20
                    ax1.set_xticks([1, 5, 10, 15, 20])
                    
                    # Horizontal line at 0% (no change)
                    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Add accuracy line to the same axis
                    sns.lineplot(x="bootstrap_rounds", y="rel_accuracy", data=df, marker="s", 
                                 color=color_accuracy, ax=ax1, label="AUC", 
                                 linestyle="--", linewidth=2)
                    
                    # Title
                    ax1.set_title(titles[i])
                    
                    # Remove default legends created by seaborn
                    if ax1.get_legend():
                        ax1.get_legend().remove()
                    
                    # Add grid for better readability
                    ax1.grid(True, alpha=0.3)
                
                # Hide unused axes in this row
                for i in range(len(bootstrap_dfs), n_cols_features):
                    axes_features[current_row, i].axis('off')
                
                # Add row title
                fig_features.text(0.02, 0.9 - (current_row * 0.3), "Bootstrap Rounds", fontsize=14, fontweight='bold')
            
            # Create a common legend for features plot
            # Get handles and labels from the first non-empty plot
            legend_handles = []
            legend_labels = []
            
            if len(self.multitesting_data_list) > 0:
                lines, labels = axes_features[0, 0].get_legend_handles_labels()
                legend_handles = lines
                legend_labels = labels
            elif len(self.buffer_data_list) > 0:
                lines, labels = axes_features[0, 0].get_legend_handles_labels()
                legend_handles = lines
                legend_labels = labels
            elif len(self.bootstrap_data_list) > 0:
                lines, labels = axes_features[0, 0].get_legend_handles_labels()
                legend_handles = lines
                legend_labels = labels
            
            fig_features.legend(legend_handles, legend_labels, loc='upper center', 
                       ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.98),
                       frameon=True, facecolor='white', edgecolor='black',
                       handlelength=3)
            
            # Tight layout with space for the legend
            fig_features.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save features plot
            plt.figure(fig_features.number)
            plt.savefig(Directory.FIGURES_DIR / save_path_features, dpi=300, bbox_inches="tight")


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

        support_datasets = []
        titles = []

        for dataset in data_copy_support[dataset_col].unique():

            data_copy_sub = data_copy_support[data_copy_support[dataset_col] == dataset]

            support_impact_data = SupportThresholdImpactData(
                min_support=data_copy_sub[rel_support],
                runtime=data_copy_sub[metric_duration],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            support_datasets.append(support_impact_data)
            titles.append(title)

        if len(support_datasets) > 0:

            plotter = SupportThresholdImpactPlot(
                data_list=support_datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('multitest', na=False)
        data_copy_multitest = data_copy[mask]

        multitest_datasets = []
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
                

            multitest_datasets.append(multitesting_data)
            titles.append(title)

        if len(multitest_datasets) > 0:

            plotter = MultiTestingImpactPlot(
                data_list=multitest_datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('buffer', na=False)
        data_copy_buffer = data_copy[mask]

        buffer_datasets = []
        titles = []

        for dataset in data_copy_buffer[dataset_col].unique():

            data_copy_sub = data_copy_buffer[data_copy_buffer[dataset_col] == dataset]

            buffer_data = BufferImpactData(
                buffer=data_copy_sub['criterion_buffer'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            buffer_datasets.append(buffer_data)
            titles.append(title)

        if len(buffer_datasets) > 0:

            plotter = BufferImpactPlot(
                data_list=buffer_datasets
            )

            plotter.plot_multiple(
                titles=titles
            )

        mask = exp_names.str.contains('bootstrap', na=False)
        data_copy_bootstrap = data_copy[mask]

        bootstrap_datasets = []
        titles = []

        for dataset in data_copy_bootstrap[dataset_col].unique():

            data_copy_sub = data_copy_bootstrap[data_copy_bootstrap[dataset_col] == dataset]

            bootstrap_data = BootstrapRoundsData(
                bootstrap_rounds=data_copy_sub['params.preprocess.params.extractor.params.bootstrap_repetitions'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v]
            )

            title = f"{dataset.split('.')[2].upper()}"
                
            bootstrap_datasets.append(bootstrap_data)
            titles.append(title)

        if len(bootstrap_datasets) > 0:

            plotter = BootstrapRoundsPlot(
                data_list=bootstrap_datasets
            )

            plotter.plot_multiple(
                titles=titles
            )


        all_in_one = AllInOnePlot(
            bootstrap_data_list=bootstrap_datasets,
            support_data_list=support_datasets,
            multitesting_data_list=multitest_datasets,
            buffer_data_list=buffer_datasets
        )

        all_in_one.plot(
            save_path_runtime="sensitivity_runtime_plot.pdf",
            save_path_features="sensitivity_features_plot.pdf"
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
