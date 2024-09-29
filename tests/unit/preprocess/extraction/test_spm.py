import unittest
import pandas as pd

from src.preprocess.extraction.spm import PrefixSpan
from src.preprocess.util.types import FrequentPattern, FrequentPatternWithConfidence
from src.preprocess.util.datasets import DatasetUniqueRulesSchema
from src.util.datasets import Dataset, DatasetSchema


class TestPrefixSpan(unittest.TestCase):

    def setUp(self) -> None:
        
        # Example dataset: a list of sequences
        self.database = [
            ['A', 'B', 'C', 'D'],  # 0
            ['A', 'B', 'C', 'D'],  # 1
            ['A', 'C', 'B', 'E'],  # 0
            ['A', 'B', 'C', 'E'],  # 1
            ['B', 'C', 'E'],       # 1
            ['A', 'B', 'D', 'E']   # 1
        ]

        self.classes = [
            0, 1, 0, 1, 1, 1
        ]

        self.prefix_df = Dataset.from_observations(
            sequences=self.database,
            classes=self.classes,
        )

    def test_get_item_counts(self):

        prefixspan = PrefixSpan()

        sequences = self.prefix_df.get_sequences()

        database = [sequence.sequence_values for sequence in sequences]
        classes = [sequence.class_value for sequence in sequences]

        item, item_neg, item_pos = prefixspan.get_item_counts(database, classes)

        for el in [item, item_neg, item_pos]:
            with self.subTest(msg=f'item: {el}'):
                self.assertIsInstance(el, dict)

    def test_get_item_counts_without_classes(self):

        prefixspan = PrefixSpan()

        sequences = self.prefix_df.get_sequences()

        database = [sequence.sequence_values for sequence in sequences]

        item, item_neg, item_pos = prefixspan.get_item_counts(database, [None for _ in range(len(database))])

        for el in [item, item_neg, item_pos]:
            with self.subTest(msg=f'item: {el}'):
                self.assertIsInstance(el, dict)

    def test_get_frequent_patterns(self):

        prefixspan = PrefixSpan(
            min_support_abs=2,
        )

        freq_patterns = prefixspan.get_frequent_patterns(
            self.prefix_df.get_sequences()
        )

        self.assertIsInstance(freq_patterns, list)
        self.assertTrue(all([isinstance(el, FrequentPattern) for el in freq_patterns]))

        freq_pattern_first = FrequentPattern(
            sequence_values=['A'],
            support=5,
            support_pos=3, 
            support_neg=2
        )

        freq_pattern_last = FrequentPattern(
            sequence_values=['A', 'B', 'C', 'D'], 
            support=2, 
            support_pos=1, 
            support_neg=1
        )

        self.assertTrue(freq_pattern_first == freq_patterns[0])
        self.assertTrue(freq_pattern_last == freq_patterns[-1])
        self.assertTrue(len(freq_patterns) == 22)

    def test_get_frequent_patterns_without_classes(self):

        prefixspan_df = Dataset.from_observations(
            sequences=self.database
        )

        prefixspan = PrefixSpan(
            min_support_abs=2,
        )

        freq_patterns = prefixspan.get_frequent_patterns(
            prefixspan_df.get_sequences()
        )

        self.assertIsInstance(freq_patterns, list)
        self.assertTrue(all([isinstance(el, FrequentPattern) for el in freq_patterns]))

        freq_pattern_first = FrequentPattern(
            sequence_values=['A'],
            support=5
        )

        freq_pattern_last = FrequentPattern(
            sequence_values=['A', 'B', 'C', 'D'], 
            support=2
        )

        self.assertTrue(freq_pattern_first == freq_patterns[0])
        self.assertTrue(freq_pattern_last == freq_patterns[-1])
        self.assertTrue(len(freq_patterns) == 22)

    def test_get_frequent_patterns_with_confidence(self):

        prefixspan = PrefixSpan(
            min_support_abs=2,
        )

        freq_patterns = [
            FrequentPattern(sequence_values=['A'], support=5, support_pos=3, support_neg=2),
            FrequentPattern(sequence_values=['A', 'B'], support=4, support_pos=2, support_neg=2),
            FrequentPattern(sequence_values=['A', 'B', 'C'], support=3, support_pos=1, support_neg=1),
            FrequentPattern(sequence_values=['D'], support=2, support_pos=1, support_neg=1),
        ]

        freq_patterns = prefixspan.get_frequent_patterns_with_confidence(
            freq_patterns
        )

        self.assertIsInstance(freq_patterns, list)
        self.assertTrue(all([isinstance(el, FrequentPatternWithConfidence) for el in freq_patterns]))


        # patterns: A -> B & A, B --> C & A --> B, C
        self.assertTrue(len(freq_patterns) == 3)
        self.assertTrue(freq_patterns[0].confidence_pos is not None)

    def test_get_frequent_patterns_with_confidence_without_classes(self):

        prefixspan = PrefixSpan(
            min_support_abs=2,
        )

        freq_patterns = [
            FrequentPattern(sequence_values=['A'], support=5, support_pos=None, support_neg=None),
            FrequentPattern(sequence_values=['A', 'B'], support=4, support_pos=None, support_neg=None),
            FrequentPattern(sequence_values=['A', 'B', 'C'], support=3, support_pos=None, support_neg=None),
            FrequentPattern(sequence_values=['D'], support=2, support_pos=None, support_neg=None),
        ]

        freq_patterns = prefixspan.get_frequent_patterns_with_confidence(
            freq_patterns
        )

        self.assertIsInstance(freq_patterns, list)
        self.assertTrue(all([isinstance(el, FrequentPatternWithConfidence) for el in freq_patterns]))


        # patterns: A -> B & A, B --> C & A --> B, C
        self.assertTrue(len(freq_patterns) == 3)
        self.assertTrue(freq_patterns[0].confidence_pos is None)

    def test_min_support(self):

        prefixspan = PrefixSpan(
            min_support_abs=3
        )

        patterns = prefixspan.get_frequent_patterns(
            self.prefix_df.get_sequences()
        )

        patterns_df = pd.DataFrame([el.model_dump() for el in patterns])

        sequence_values = patterns_df['sequence_values'].to_list()

        self.assertTrue(len(sequence_values) == 16)

        self.assertIn(['A'], sequence_values)
        self.assertIn(['A', 'B'], sequence_values)
        self.assertIn(['B', 'E'], sequence_values)
        self.assertIn(['A', 'B', 'C'], sequence_values)
        self.assertNotIn(['D', 'E'], sequence_values)
        self.assertNotIn(['B', 'C', 'E'], sequence_values)
        
    def test_max_pattern_length(self):

        prefixspan = PrefixSpan(
            max_sequence_length=1
        )

        patterns = prefixspan.get_frequent_patterns(
            self.prefix_df.get_sequences()
        )

        patterns_df = pd.DataFrame([el.model_dump() for el in patterns])
        sequence_values = patterns_df['sequence_values'].to_list()

        self.assertTrue(len(patterns) == 5)
        self.assertIn(['A'], sequence_values)
        self.assertNotIn(['B', 'E'], sequence_values)

    def test_summarise_patterns_in_dataframe(self):

        prefixspan = PrefixSpan()

        result = prefixspan.summarise_patterns_in_dataframe(
            self.prefix_df.raw_data
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(DatasetUniqueRulesSchema.delta_confidence in result.columns)

        result = prefixspan.summarise_patterns_in_dataframe(
            self.prefix_df.raw_data.drop(
                columns=[DatasetSchema.class_column]
            )
        )

        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(DatasetUniqueRulesSchema.delta_confidence in result.columns)

    def test__encode_train(self):

        feat_alg = PrefixSpan()

        result = feat_alg._encode_train(
            data=self.prefix_df.raw_data
        )

        # assert type
        self.assertIsInstance(result, dict)

        # assert keys
        self.assertIn('data', result)
        self.assertIn('rules', result) 
        result_df = result['data']
        self.assertIsInstance(result_df, pd.DataFrame)

        # check if class column is present
        self.assertTrue(DatasetSchema.class_column in result_df.columns)

        # check if dataframe only contains binary results
        self.assertTrue(all([el in [0, 1] for el in result_df.values.flatten()]))

        # check output for individual rules
        rules = {
            'A_B_C': [1, 1, 0, 1, 0, 0],
            'D_E': [0, 0, 0, 0, 0, 1],
            'C_B': [0, 0, 1, 0, 0, 0],
        }

        for rule, expected in rules.items():
            with self.subTest(msg=f'rule: {rule}'):
                self.assertTrue(all([el == expected[i] for i, el in enumerate(result_df[rule].values)]))

    def test__encode_test(self):

        feat_alg = PrefixSpan()

        result = feat_alg._encode_train(
            data=self.prefix_df.raw_data
        )

        result['data'] = self.prefix_df.raw_data.drop(
            columns=[DatasetSchema.class_column]
        )

        result = feat_alg._encode_test(**result)

        # assert type
        self.assertIsInstance(result, dict)

        # assert keys
        self.assertIn('data', result)
        self.assertNotIn('rules', result) 
        result_df = result['data']
        self.assertIsInstance(result_df, pd.DataFrame)

        # check if class column is present
        self.assertTrue(DatasetSchema.class_column in result_df.columns)

        # check if dataframe only contains binary results
        self.assertTrue(all([el in [0, 1] for el in result_df.drop(columns=[DatasetSchema.class_column]).values.flatten()]))

        # check output for individual rules
        rules = {
            'A_B_C': [1, 1, 0, 1, 0, 0],
            'D_E': [0, 0, 0, 0, 0, 1],
            'C_B': [0, 0, 1, 0, 0, 0],
        }

        for rule, expected in rules.items():
            with self.subTest(msg=f'rule: {rule}'):
                self.assertTrue(all([el == expected[i] for i, el in enumerate(result_df[rule].values)]))
