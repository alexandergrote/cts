import numpy as np

from typing import Tuple
from scipy.stats import chi2_contingency, fisher_exact


def compute_chi_square(rule_pos: int, rule_neg: int, non_rule_pos: int, non_rule_neg: int) -> Tuple[float, float]:
    
    contingency_matrix = np.array([[rule_pos, rule_neg], [non_rule_pos, non_rule_neg]])

    try:
        chi2, p, _, _ = chi2_contingency(contingency_matrix)
    except ValueError:
        chi2, p = float('nan'), float('nan')  # if any issue (e.g., division by zero)

    return chi2, p


def compute_phi_coefficient(rule_pos: int, rule_neg: int, non_rule_pos: int, non_rule_neg: int) -> float:
    """
    Compute the phi coefficient (correlation coefficient for binary variables) from a 2x2 contingency table.
    
    Parameters:
    ----------
    rule_pos : int
        Count of positive instances covered by the rule
    rule_neg : int
        Count of negative instances covered by the rule
    non_rule_pos : int
        Count of positive instances not covered by the rule
    non_rule_neg : int
        Count of negative instances not covered by the rule
    
    Returns:
    -------
    float
        The phi coefficient value
    """
    
    # Calculate each term in the phi coefficient formula
    n = rule_pos + rule_neg + non_rule_pos + non_rule_neg
    
    # Prevent division by zero
    if n == 0:
        return float('nan')
    
    # Calculate row and column sums
    row1_sum = rule_pos + rule_neg
    row2_sum = non_rule_pos + non_rule_neg
    col1_sum = rule_pos + non_rule_pos
    col2_sum = rule_neg + non_rule_neg
    
    # Check for zero denominators to avoid division by zero
    if row1_sum == 0 or row2_sum == 0 or col1_sum == 0 or col2_sum == 0:
        return float('nan')
    
    # Compute phi coefficient: (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    numerator = (rule_pos * non_rule_neg) - (rule_neg * non_rule_pos)
    denominator = np.sqrt(row1_sum * row2_sum * col1_sum * col2_sum)
    
    phi = numerator / denominator
    
    return phi

def compute_fisher(rule_pos: int, rule_neg: int, non_rule_pos: int, non_rule_neg: int) -> Tuple[float, float]:
    """
    Compute Fisher's exact test for a 2x2 contingency table.
    
    Args:
        rule_pos: Number of positive cases where the rule applies
        rule_neg: Number of negative cases where the rule applies
        non_rule_pos: Number of positive cases where the rule does not apply
        non_rule_neg: Number of negative cases where the rule does not apply
        
    Returns:
        Tuple containing the odds ratio and p-value
    """
    contingency_matrix = np.array([[rule_pos, rule_neg], [non_rule_pos, non_rule_neg]])
    
    try:
        odds_ratio, p_value = fisher_exact(contingency_matrix)
    except ValueError:
        odds_ratio, p_value = float('nan'), float('nan')  # if any issue occurs
        
    return odds_ratio, p_value


def compute_leverage(num_observations: int, num_pos: int, num_seq: int, num_pos_seq: int):
    """"
    With this function, we calculate the leverage of a dataset.
    It is also known as 1-quality for subgroup discovery and can be used to measure the importance of a subgroup.
    One downside is that leverage is also known to slightly favors itemsets with larger support.

    parameters:
       num_observations: int, the total number of observations in the dataset.
       num_pos: int, the number of positive observations in the dataset.
       num_seq: int, the frequency of a target sequence in the dataset.
       num_pos_seq: int, the frequency of a target sequence with positive labels. 

    See this paper for more information: https://www.vldb.org/pvldb/vol17/p2668-vandin.pdf

    """

    assert num_observations > 0, "Number of observations must be greater than 0"
    assert num_pos >= 0, "Number of positive observations must be non-negative"
    assert num_seq >= 0, "Number of sequences must be non negative"
    assert num_observations >= num_seq, "Number of observations must be greater than or equal to the number of sequences"
    assert num_pos_seq >= 0, "Number of positive observations in the target sequence must be non-negative"
    assert num_pos_seq <= num_pos, "Number of positive observations in the target sequence"
    assert num_pos_seq <= num_seq, "Number of positive observations in the target sequence must be less than or equal to the number of sequences"

    joint_proba = num_pos_seq / num_observations
    class_proba = num_pos / num_observations
    seq_proba = num_seq / num_observations

    return joint_proba - class_proba * seq_proba
