import numpy as np

class ConfidenceCalculator:

    @staticmethod
    def calculate_confidence(support_antecedent: int, support_antecedent_and_consequent: int) -> float:
        
        assert support_antecedent >= support_antecedent_and_consequent, f"support antecedent {support_antecedent} should be greater or equal to support antecedent and consequent {support_antecedent_and_consequent}"
        assert support_antecedent >= 0, f"support antecedent {support_antecedent} should be greater than 0"

        if support_antecedent == 0:
            return 0

        return support_antecedent_and_consequent / support_antecedent


class EntropyCalculator:

    @staticmethod
    def calculate_entropy(probability: float) -> float:

        if probability == 0 or probability == 1:
            return 0

        return -probability * np.log2(probability) - (1-probability) * np.log2(1-probability)

