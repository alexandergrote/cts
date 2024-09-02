import pandas as pd
import unittest

from src.preprocess.extraction.ts_features import Dataset, \
    AnnotatedSequence, PrefixSpan, FrequentPatternWithConfidence, \
    FrequentPattern, SPMFeatureSelector


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:

        self.id_column = 'id_column'
        self.time_column = 'time_column'
        self.event_column = 'event_column'
        self.class_column = 'class_column'

        self.raw_data = pd.DataFrame({
            self.id_column: ['a', 'a', 'a', 'b', 'b'],
            self.time_column: [i for i in range(5)],
            self.event_column: [1,2,3,5,4],
            self.class_column: [1, 0, 1, 0, 1]
        })

    def test_get_sequences(self):

        dataset = Dataset(
            id_column=self.id_column,
            event_column=self.event_column,
            time_column=self.time_column,
            class_column=self.class_column,
            raw_data=self.raw_data
        )

        sequences = dataset.get_sequences()

        self.assertIsInstance(sequences, list)
        self.assertTrue(all([isinstance(el, AnnotatedSequence) for el in sequences]))
        self.assertEqual([el.sequence_values for el in sequences], [['1','2', '3'], ['5','4']])
        self.assertEqual([el.id_value for el in sequences], ['a', 'b'])
        
    def test_get_sequences_without_classes(self):

        data_copy = self.raw_data.copy(deep=True)
        data_copy.drop(columns=self.class_column, inplace=True)

        dataset = Dataset(
            id_column=self.id_column,
            event_column=self.event_column,
            time_column=self.time_column,
            raw_data=self.raw_data
        )

        sequences = dataset.get_sequences()

        self.assertIsInstance(sequences, list)
        self.assertTrue(all([isinstance(el, AnnotatedSequence) for el in sequences]))
        self.assertEqual([el.sequence_values for el in sequences], [['1', '2', '3'], ['5', '4']])
        self.assertEqual([el.id_value for el in sequences], ['a', 'b'])


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

        print(result)

    def test_execute(self):

        prefixspan = PrefixSpan()

        patterns = prefixspan.execute(self.prefix_df.raw_data)

        for pattern in patterns:
            with self.subTest(msg=f'patterns: {pattern}'):
                self.assertIsInstance(pattern, FrequentPatternWithConfidence)

        
class TestSPMFeatureSelection(unittest.TestCase):

    def setUp(self) -> None:
        
        # Example dataset: a list of sequences
        self.database = [
            ['A', 'B', 'C', 'D'],
            ['A', 'B', 'C', 'D'],
            ['A', 'C', 'B', 'E'],
            ['A', 'B', 'C', 'E'],
            ['B', 'C', 'E'],
            ['A', 'B', 'D', 'E']
        ]

        self.classes = [
            0, 1, 0, 1, 1, 1
        ]

        self.prefix_df = Dataset.from_observations(
            sequences=self.database,
            classes=self.classes,
        )

        self.prefixspan_config = {
            'class_name': 'src.preprocess.extraction.ts_features.PrefixSpan',
            'params': {}
        }

    def test__bootstrap(self):

        feat_alg = SPMFeatureSelector(
            prefixspan_config=self.prefixspan_config
        )

        patterns = feat_alg._bootstrap(data=self.prefix_df.raw_data)
    
        for pattern in patterns:
            self.assertIsInstance(pattern, FrequentPatternWithConfidence)

    def test_encode_train(self):

        feat_alg = SPMFeatureSelector(
            prefixspan_config=self.prefixspan_config
        )

        result = feat_alg._encode_train(
            data=self.prefix_df.raw_data
        )

        # assert type
        self.assertIsInstance(result, dict)

        # assert keys
        assert 'data' in result.keys()
        result_df = result['data']
        self.assertIsInstance(result_df, pd.DataFrame)

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

    def test_encode_test(self):

        feat_alg = SPMFeatureSelector(
            prefixspan_config=self.prefixspan_config
        )

        kwargs = feat_alg._encode_train(
            data=self.prefix_df.raw_data
        )

        kwargs["data"] = self.prefix_df.raw_data

        result = feat_alg._encode_test(**kwargs)

        # assert type
        self.assertIsInstance(result, dict)

        result_df = result['data']
        self.assertIsInstance(result_df, pd.DataFrame)

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


if __name__ == '__main__':
    unittest.main()