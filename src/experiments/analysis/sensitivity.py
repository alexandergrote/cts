import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
            

from abc import ABC, abstractmethod
from pandas.api.types import CategoricalDtype
from typing import List, Optional, Dict, Any, Tuple, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator

from src.experiments.analysis.base import BaseAnalyser
from src.util.constants import Directory

# Set black and white style
#sns.set_style('white')
#plt.rcParams['font.size'] = 14
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'black'])


class BaseData(ABC):

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        raise ValueError("This method should be implemented by subclasses")

class MixinData(BaseModel):
    dataset_name: str = Field(description="Name of the dataset")
    accuracy: List[float] = Field(description="Classification accuracy for each threshold")
    number_of_features: List[int] = Field(description="Number of features selected for each threshold")
    runtime: List[float] = Field(description="Runtime in seconds for each threshold")

    def get_threshold_name(self) -> str:

        list_of_annotations = list(self.__annotations__.keys())
        if len(list_of_annotations) != 1:
            raise ValueError("There should be exactly one annotation defined.")
        return list_of_annotations[0]
        
    def normalize_df(self, data: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:

        df_copy = data.copy(deep=True)

        # Referenzwerte (Mittelwert aller "multitesting == False")
        ref_features = df_copy.loc[mask, 'number_of_features'].mean()
        ref_accuracy = df_copy.loc[mask, 'accuracy'].mean()
        ref_runtime = df_copy.loc[mask, 'runtime'].mean()
        
        # Berechne relative Änderungen in Prozent
        df_copy['rel_number_of_features'] = (df_copy['number_of_features'] / ref_features - 1) * 100
        df_copy['rel_accuracy'] = (df_copy['accuracy'] / ref_accuracy - 1) * 100
        df_copy['rel_runtime'] = (df_copy['runtime'] / ref_runtime - 1) * 100
        return df_copy
        


class MultiTestingImpactData(MixinData, BaseData):
    multitesting: List[str] = Field(description="Boolean indication of whether multitesting was applied or not")

    def to_df(self) -> pd.DataFrame:

        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'dataset_name': self.dataset_name,
            'multitesting': self.multitesting,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy,
            'runtime': self.runtime,
        })
        
        no_correction_mask = df['multitesting'] == str(False)
        
        df = df.replace({str(True): 'With Correction', str(False): 'No Correction'})

           
        # Sortiere die Kategorien, damit "With Correction" immer links und "No Correction" immer rechts ist
        df['multitesting'] = pd.Categorical(
            df['multitesting'], 
            categories=["With Correction", "No Correction"], 
            ordered=True
        )
        
        return self.normalize_df(data=df, mask=no_correction_mask)


class SupportThresholdImpactData(MixinData, BaseData):
    min_support: List[float] = Field(description="Minimum support threshold values")
    
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
        required_cols = ['min_support', 'runtime', 'accuracy', ]
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
            'dataset_name': self.dataset_name,
            'min_support': self.min_support,
            'runtime': self.runtime,
            'accuracy': self.accuracy,
            'number_of_features': self.number_of_features
        })
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest min_support as reference
        min_support_value = df['min_support'].min()
        min_support_mask = df['min_support'] == min_support_value
        
        return self.normalize_df(data=df, mask=min_support_mask)

class MaxSequenceImpactData(MixinData, BaseData):
    max_sequence: List[int] = Field(description="Max sequence length values")
    
    # Field validators (V2 style)
    @field_validator('max_sequence')
    @classmethod
    def validate_min_support(cls, v: List[float]) -> List[float]:
        if any(x < 2 for x in v):
            raise ValueError("All minimum support values must not be smaller than 2")
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
        lists = [self.max_sequence, self.runtime, self.accuracy]
        if len(set(len(lst) for lst in lists)) > 1:
            raise ValueError("All lists must have the same length")
        return self
    
    
    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'dataset_name': self.dataset_name,
            'max_sequence': self.max_sequence,
            'runtime': self.runtime,
            'accuracy': self.accuracy,
            'number_of_features': self.number_of_features
        })
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest min_support as reference
        max_sequence_value = df['max_sequence'].min()
        max_sequence_mask = df['max_sequence'] == max_sequence_value
        
        return self.normalize_df(data=df, mask=max_sequence_mask)

    


class BufferImpactData(MixinData, BaseData):

    buffer: List[float] = Field(description="Criterion buffer values")
    
    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'dataset_name': self.dataset_name,
            'buffer': self.buffer,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy,
            'runtime': self.runtime,
        })
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest buffer value as reference
        min_buffer_value = df['buffer'].min()
        min_buffer_mask = df['buffer'] == min_buffer_value
        
        return self.normalize_df(data=df, mask=min_buffer_mask)


class BootstrapRoundsData(MixinData, BaseData):
    bootstrap_rounds: List[int] = Field(description="Number of bootstrap rounds")
    
    def to_df(self) -> pd.DataFrame:
        """Convert the data to a DataFrame with validation"""
        df = pd.DataFrame({
            'dataset_name': self.dataset_name,
            'bootstrap_rounds': self.bootstrap_rounds,
            'number_of_features': self.number_of_features,
            'accuracy': self.accuracy,
            'runtime': self.runtime
        })
        
        # Calculate relative changes
        # Use the mean of all data points with the smallest bootstrap_rounds as reference
        min_rounds_value = df['bootstrap_rounds'].min()
        min_rounds_mask = df['bootstrap_rounds'] == min_rounds_value
        
        return self.normalize_df(data=df, mask=min_rounds_mask)
        

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
        #sns.set_style(style)
        
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
        #sns.set_style(style)
        
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
        #sns.set_style(style)
        
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
        #sns.set_style(style)
        
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
    max_sequence_data_list: List[MaxSequenceImpactData]

    def plot(self, 
             titles: Optional[List[str]] = None,
             save_path: str = "sensitivity_all_in_one_plot.pdf",
             figsize: Tuple[int, int] = (18, 24),  # Taller figure for 4 rows
             style: str = "whitegrid",
             color_runtime: str = "black",
             color_accuracy: str = "black",
             color_num_features: str = "black",
             ) -> None:
        """
        Create a combined plot with all sensitivity analyses in one figure.
        Each row represents one type of plot.
        
        Args:
            titles: List of titles for each column (dataset). If None, default titles are used.
            save_path: Path to save the figure
            figsize: Figure size (width, height) in inches
            style: Seaborn style to use
            color_runtime: Color for runtime/number of features line
            color_accuracy: Color for accuracy line
        """
        # Check if we have data for at least one plot type
        if (len(self.support_data_list) == 0 and 
            len(self.multitesting_data_list) == 0 and 
            len(self.buffer_data_list) == 0 and 
            len(self.bootstrap_data_list) == 0 and 
            len(self.max_sequence_data_list) == 0):
            raise ValueError("At least one dataset must be provided for at least one plot type")
        
        # Determine the number of columns (datasets)
        n_cols = max(
            len(self.support_data_list),
            len(self.multitesting_data_list),
            len(self.buffer_data_list),
            len(self.bootstrap_data_list),
            len(self.max_sequence_data_list)
        )
        
        if n_cols == 0:
            raise ValueError("No datasets provided")

        data_lists = [self.support_data_list, self.max_sequence_data_list, self.multitesting_data_list, self.buffer_data_list, self.bootstrap_data_list]
        df_to_plot = []
        for data_list in data_lists:
            assert isinstance(data_list, list), "All elements in data_lists must be lists"
            for el in data_list:
                assert isinstance(el, BaseData), "All elements in data_lists must be instances of BaseData"
                assert isinstance(el, MixinData), "All elements in data_lists must be instances of MixinData"
                
                df = el.to_df()
                df.rename(columns={el.get_threshold_name(): 'x_axis'}, inplace=True)
                df['scenario'] = el.__class__.__name__

                print(df.head())
            
                # Wenn es sich um Multitesting-Daten handelt, behalte die in to_df() festgelegte Reihenfolge bei
                if el.__class__.__name__ == 'MultiTestingImpactData':
                    pass
            
                df_melt = df.melt(id_vars=['dataset_name','scenario', 'x_axis'], value_vars=['rel_number_of_features', 'rel_accuracy', 'rel_runtime'])
                df_to_plot.append(df_melt)

            
        df_to_plot = pd.concat(df_to_plot, ignore_index=True)

        grey_palette = sns.color_palette(['grey'] * df_to_plot['variable'].nunique())  # Adjust the number of grey tones based on the unique values in your hue column
        df_to_plot['dataset_name'] = df_to_plot['dataset_name'].apply(lambda x: x.split('.')[-1])
        
        df_to_plot.replace({

            'rel_number_of_features': 'Number of Features',
            'rel_accuracy': 'AUC',
            'rel_runtime': 'Runtime',
            'ChurnDataloader': 'Churn',
            'MalwareDataloader': 'Malware',
            'DataLoader': 'Synthetic',
            'SupportThresholdImpactData': 'Min Support',
            'MaxSequenceImpactData': 'Sequence Length',
            'MultiTestingImpactData': 'Multitesting',
            'BufferImpactData': 'Buffer',
            'BootstrapRoundsData': 'Bootstrap Rounds',
                       
        }, inplace=True)

        y_axis = 'Relative Change [%]'
        x_axis = 'Parameter Value'
        df_to_plot.rename(columns={'x_axis': x_axis, 'value': y_axis}, inplace=True)

        # Define markers for different categories in your hue column
        markers = {
            'Number of Features': 'o', 
            'AUC': 'D',
            'Runtime': 's',
        }

        mpl.rcParams['lines.markersize'] = 10

        sns.set(font_scale=1.5)
        sns.set_style('white')

        custom_order = [
            '0.0', '0.05', '0.1', '0.15', '0.2', '0.25',
            'No Correction', 'With Correction',
            '2', '3', '4',
            '10', '15', '20'
        ]

        cat_type = CategoricalDtype(categories=custom_order, ordered=True)

        df_to_plot[x_axis] = df_to_plot[x_axis].astype(str).astype(cat_type)
        
        g = sns.FacetGrid(df_to_plot, col='dataset_name', row='scenario', sharey=False, height=4, sharex=False, despine=False)

        g.map_dataframe(
            sns.lineplot, 
            x=x_axis, 
            y=y_axis,
            hue='variable',
            palette=grey_palette,
            style='variable',
            markers=markers,
            errorbar=None,
            dashes=True,
        )

        for ax in g.axes.flat:
            for line in ax.lines:
                line.set_markersize(15)  # Set your desired marker size

        
        # Here we're using only '{col_name}' to display just the value of the column variable
        g.set_titles("{row_name} | {col_name}")

        handles, labels = g.axes.flat[0].get_legend_handles_labels()

        # Remove any duplicate labels/handles
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):

            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        # Füge schwarze Rahmen um alle Plots hinzu
        for ax_row in g.axes:
            for ax in ax_row:
                # Füge einen schwarzen Rahmen um jeden Plot hinzu
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1)

        # Draw the unique legend
        g.fig.legend(handles=unique_handles, loc='upper center', labels=unique_labels, bbox_to_anchor=(0.5, 1.05), ncol=len(unique_labels))

        # Optionally, adjust the figure to make room for the legend if needed
        g.fig.subplots_adjust(top=0.9)

        plt.tight_layout()        
        
        # Save if path is provided
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)


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
        max_sequence = 'max_sequence_length'
        n_features_selected = 'metrics.n_features_selected'
        criterion_buffer = 'criterion_buffer'
        
        # create mapping for renaming
        mapping = {
            'params.fetch_data.class_name': dataset_col,
            'metrics.feature_selection_duration': metric_duration,
            'metrics.feature_selection_max_memory': metric_memory,
            'params.preprocess.params.extractor.params.prefixspan_config.params.min_support_rel': rel_support,
            'params.preprocess.params.extractor.params.prefixspan_config.params.max_sequence_length': max_sequence,
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
                dataset_name=dataset,
                min_support=data_copy_sub[rel_support],
                runtime=data_copy_sub[metric_duration],
                accuracy=data_copy_sub[metric_col_auc_v],
                number_of_features=data_copy_sub['N Features Selected']
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

        # max sequence datasets
        mask = exp_names.str.contains('max_sequence', na=False)
        data_copy_sequence = data_copy[mask]

        sequence_datasets = []
        titles = []

        for dataset in data_copy_sequence[dataset_col].unique():

            data_copy_sub = data_copy_sequence[data_copy_sequence[dataset_col] == dataset]

            support_impact_data = MaxSequenceImpactData(
                dataset_name=dataset,
                max_sequence=data_copy_sub[max_sequence],
                runtime=data_copy_sub[metric_duration],
                accuracy=data_copy_sub[metric_col_auc_v],
                number_of_features=data_copy_sub['N Features Selected']
            )

            title = f"{dataset.split('.')[2].upper()}"
                

            sequence_datasets.append(support_impact_data)
            titles.append(title)


        # multitest datasets
        mask = exp_names.str.contains('multitest', na=False)
        data_copy_multitest = data_copy[mask]

        multitest_datasets = []
        titles = []

        for dataset in data_copy_multitest[dataset_col].unique():

            data_copy_sub = data_copy_multitest[data_copy_multitest[dataset_col] == dataset]

            # Keine Sortierung mehr nötig, da wir den Mittelwert aller "multitesting == False" verwenden
            
            multitesting_data = MultiTestingImpactData(
                dataset_name=dataset,
                multitesting=data_copy_sub['params.export.params.experiment_name'].str.contains('True', na=False).astype(str),
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v],
                runtime=data_copy_sub[metric_duration],
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
                dataset_name=dataset,
                buffer=data_copy_sub['criterion_buffer'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v],
                runtime=data_copy_sub[metric_duration]
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
                dataset_name=dataset,
                bootstrap_rounds=data_copy_sub['params.preprocess.params.extractor.params.bootstrap_repetitions'],
                number_of_features=data_copy_sub['N Features Selected'],
                accuracy=data_copy_sub[metric_col_auc_v],
                runtime=data_copy_sub[metric_duration]
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
            buffer_data_list=buffer_datasets,
            max_sequence_data_list=sequence_datasets,
        )

        all_in_one.plot()

        # Create synthetic-only plot
        synthetic_support_datasets = [d for d in support_datasets if 'synthetic' in d.dataset_name]
        synthetic_buffer_datasets = [d for d in buffer_datasets if 'synthetic' in d.dataset_name]
        synthetic_bootstrap_datasets = [d for d in bootstrap_datasets if 'synthetic' in d.dataset_name]
        synthetic_multitest_datasets = [d for d in multitest_datasets if 'synthetic' in d.dataset_name]
        synthetic_sequence_datasets = [d for d in sequence_datasets if 'synthetic' in d.dataset_name]
        
        if any([synthetic_support_datasets, synthetic_buffer_datasets, synthetic_bootstrap_datasets, 
                synthetic_multitest_datasets, synthetic_sequence_datasets]):
            self.plot_synthetic_only(
                support_data_list=synthetic_support_datasets,
                buffer_data_list=synthetic_buffer_datasets,
                bootstrap_data_list=synthetic_bootstrap_datasets,
                multitesting_data_list=synthetic_multitest_datasets,
                max_sequence_data_list=synthetic_sequence_datasets
            )

    def plot_synthetic_only(self,
                           support_data_list: List[SupportThresholdImpactData] = [],
                           multitesting_data_list: List[MultiTestingImpactData] = [],
                           buffer_data_list: List[BufferImpactData] = [],
                           bootstrap_data_list: List[BootstrapRoundsData] = [],
                           max_sequence_data_list: List[MaxSequenceImpactData] = [],
                           save_path: str = "sensitivity_synthetic_only.pdf",
                           figsize: Tuple[int, int] = (20, 5)):
        """
        Create a plot showing all threshold plots for the synthetic dataset in one row.
        """
        # Check if we have any data
        if not any([support_data_list, multitesting_data_list, buffer_data_list, 
                   bootstrap_data_list, max_sequence_data_list]):
            return
            
        # Prepare data for plotting
        df_to_plot = []
        
        # Process each data type
        data_lists = [
            ('Min Support', support_data_list),
            ('Maximum Sequence Length', max_sequence_data_list),
            ('Multitesting', multitesting_data_list),
            ('Minimum Effect Size', buffer_data_list),
            ('Bootstrap Rounds', bootstrap_data_list)
        ]
        
        for scenario_name, data_list in data_lists:
            if not data_list:
                continue
                
            for el in data_list:
                df = el.to_df()
                df.rename(columns={el.get_threshold_name(): 'x_axis'}, inplace=True)
                df['scenario'] = scenario_name
                
                df_melt = df.melt(id_vars=['dataset_name', 'scenario', 'x_axis'], 
                                  value_vars=['rel_number_of_features', 'rel_accuracy', 'rel_runtime'])
                df_to_plot.append(df_melt)
        
        if not df_to_plot:
            return
            
        df_to_plot = pd.concat(df_to_plot, ignore_index=True)
        
        # Prepare plot settings
        grey_palette = sns.color_palette(['grey'] * df_to_plot['variable'].nunique())
        
        df_to_plot.replace({
            'rel_number_of_features': 'Number of Features',
            'rel_accuracy': 'AUC',
            'rel_runtime': 'Runtime',
        }, inplace=True)
        
        y_axis = 'Relative Change [%]'
        x_axis = 'Thresholds'
        df_to_plot.rename(columns={'x_axis': x_axis, 'value': y_axis}, inplace=True)
        
        # Define markers
        markers = {
            'Number of Features': 'o', 
            'AUC': 'D',
            'Runtime': 's',
        }
        
        # Set plot style
        sns.set(font_scale=1.5)
        sns.set_style('white')
        
        # Create categorical order for thresholds
        custom_order = [
            '0.0', '0.05', '0.1', '0.15', '0.2', '0.25',
            'No Correction', 'With Correction',
            '2', '3', '4',
            '1', '5', '10', '15', '20'
        ]
        cat_type = CategoricalDtype(categories=custom_order, ordered=True)
        df_to_plot[x_axis] = df_to_plot[x_axis].astype(str).astype(cat_type)
        
        # Create plot
        g = sns.FacetGrid(df_to_plot, col='scenario', row=None, sharey=False, 
                          height=4, aspect=1.2, sharex=False, despine=False)
        
        # Dictionary für spezifische x-Achsenbeschriftungen je nach Szenario
        x_labels = {
            'Min Support': 'Relative Support Threshold',
            'Maximum Sequence Length': 'Maximum Sequence Length',
            'Multitesting': 'Multitesting',
            'Minimum Effect Size': 'Effect Size Threshold',
            'Bootstrap Rounds': 'Number of Bootstrap Rounds'
        }
        
        g.map_dataframe(
            sns.lineplot, 
            x=x_axis, 
            y=y_axis,
            hue='variable',
            palette=grey_palette,
            style='variable',
            markers=markers,
            errorbar=None,
            dashes=True,
        )
        
        # Increase marker size and set y-axis label
        for i, ax in enumerate(g.axes.flat):
            for line in ax.lines:
                line.set_markersize(10)
            
            # Add black border around each plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)
                
            # Add horizontal line at 0%
            #ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Add grid
            #ax.grid(True, alpha=0.3)
            
            # Y-Achsenbeschriftung für alle Plots anzeigen
            ax.set_ylabel(y_axis)
        
        # Set titles and x-labels based on scenario
        #g.set_titles("{col_name}")
        g.set_titles("")
        
        # Setze spezifische x-Achsenbeschriftungen für jedes Szenario
        for ax, scenario in zip(g.axes.flat, g.col_names):
            if scenario in x_labels:
                ax.set_xlabel(x_labels[scenario])
        
        # Create legend
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        g.fig.legend(handles=unique_handles, loc='upper center', 
                    labels=unique_labels, bbox_to_anchor=(0.5, 1.15), 
                    ncol=len(unique_labels))
        
        # Adjust layout
        g.fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(Directory.FIGURES_DIR / save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)


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
