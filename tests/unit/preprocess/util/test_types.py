import unittest

from src.preprocess.util.types import FrequentPatternWithConfidence, EntropyCalculator


class TestFrequentPattern(unittest.TestCase):

    def setUp(self) -> None:

        self.shared_kwargs = dict(
            antecedent=["a"],
            consequent=["b"],
            confidence=0.8,
        )

    def test_centered_inverse_entropy(self):

        freq_pattern_pos = FrequentPatternWithConfidence(
            **self.shared_kwargs,
            support=10,
            support_neg=2,
            support_pos=8,
        )

        self.assertTrue(freq_pattern_pos.centered_inverse_entropy > 0)

        freq_pattern_neg = FrequentPatternWithConfidence(
            **self.shared_kwargs,
            support=freq_pattern_pos.support,
            support_neg=freq_pattern_pos.support_pos,
            support_pos=freq_pattern_pos.support_neg
        )

        self.assertTrue(freq_pattern_neg.centered_inverse_entropy < 0)
        self.assertEqual(-freq_pattern_pos.centered_inverse_entropy, freq_pattern_neg.centered_inverse_entropy)

        # check monotonicity
        freq_pattern_pos_smaller = FrequentPatternWithConfidence(
            **self.shared_kwargs,
            support=freq_pattern_pos.support,
            support_neg=freq_pattern_pos.support_neg + 1,
            support_pos=freq_pattern_pos.support_pos - 1
        )

        self.assertTrue(freq_pattern_pos.centered_inverse_entropy > freq_pattern_pos_smaller.centered_inverse_entropy)


        freq_pattern_no_diff = FrequentPatternWithConfidence(
            **self.shared_kwargs,
            support=2,
            support_neg=1,
            support_pos=1
        )

        self.assertEqual(freq_pattern_no_diff.centered_inverse_entropy, EntropyCalculator.calculate_entropy(probability=0.5))
        self.assertTrue(freq_pattern_no_diff.centered_inverse_entropy > freq_pattern_pos_smaller.centered_inverse_entropy)


if __name__ == '__main__':

    unittest.main()