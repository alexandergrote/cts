import numpy as np
from typing import List, Literal, Optional, Tuple

from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ttest_1samp, wilcoxon

from src.util.custom_logging import console
from src.preprocess.util.datasets import DatasetRulesSchema, DatasetRules, DatasetUniqueRules, DatasetUniqueRulesSchema, DatasetAggregated, DatasetAggregatedSchema, DatasetDeltaConfidencePValues
from src.preprocess.extraction.ts_features import SPMFeatureSelector
from src.preprocess.util.datasets import DatasetUniqueRulesSchema


class SubgroupSelector(SPMFeatureSelector):

    criterion: Literal[
        DatasetUniqueRulesSchema.chi_squared, 
        DatasetUniqueRulesSchema.fisher_odds_ratio,
        DatasetUniqueRulesSchema.phi,
    ] = DatasetUniqueRulesSchema.phi


    bootstrap_repetitions: int = 1
    bootstrap_sampling_fraction: float = 1.0

    def _select_significant_greater_than_zero(self, *, data: DatasetUniqueRules, **kwargs) -> Tuple[DatasetUniqueRules, DatasetDeltaConfidencePValues]:

        """
        Conduct statistical tests to select informative rules

        Args:
            data: DatasetRules containing all rules
            **kwargs:

        Returns:

        """

        # work on copy
        data_copy = data.data.copy(deep=True)

        # keep track of p values
        p_value_col = f"{self.criterion}_p_values"

        if p_value_col in data_copy.columns:

            p_values_array = data_copy[p_value_col].apply(lambda x: x[0]).values
        
        else:
            style = "bold white on red"
            console.log(f"Column {p_value_col} not found", style=style)
            p_values_array = -1 * np.ones(len(data_copy)) 

        pvals_corrected = np.array(p_values_array)

        if self.multitesting is None:

            # exclude rules based on p value
            mask = p_values_array <= self.p_value_threshold

        else:

            _, pvals_corrected, _, _ = multipletests(p_values_array, **self.multitesting)

            mask = np.array(pvals_corrected) <= self.p_value_threshold

        result = DatasetUniqueRules(
            data=data_copy[mask]
        )

        result_p_values = DatasetDeltaConfidencePValues.create_from_unique_rules(
            data=DatasetUniqueRules(data=data_copy), pvalues=p_values_array, corrected_pvalues=pvals_corrected, mask=p_values_array < self.p_value_threshold
        )

        console.log("Number of unique rules after selection: {}".format(result.data.shape[0]))

        return result, result_p_values