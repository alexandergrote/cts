import pandas as pd
import unittest

from datetime import datetime

from src.preprocess.extraction.ts_features import Dataset, \
    AnnotatedSequence, PrefixSpanNew, FrequentPatternWithConfidence, \
    FrequentPattern, SPMFeatureSelectorNew


class TestPrefixSpanDataset(unittest.TestCase):

    def setUp(self) -> None:

        self.id_column = 'id_column'
        self.time_column = 'time_column'
        self.event_column = 'event_column'
        self.class_column = 'class_column'

        self.raw_data = pd.DataFrame({
            self.id_column: ['a', 'a', 'a', 'b', 'b'],
            self.time_column: [datetime(2020, 1, 1, 12, i) for i in range(5)],
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

class TestPrefixSpanNew(unittest.TestCase):

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

        self.timestamps = [datetime(2020, 1, 1, 12, i) for i in range(len(self.database))]

        self.id_values = ['a', 'b', 'c', 'd', 'e', 'f']

        # create dataframe entry for each element in database
        records = []

        for i, el in enumerate(self.database):
            for event in el:

                records.append({
                    'id_column': self.id_values[i],
                    'time_column': self.timestamps[i],
                    'event_column': event,
                    'class_column': self.classes[i]
                    })
                
        self.raw_data = pd.DataFrame.from_records(records)

        self.prefix_df = Dataset(
            raw_data=self.raw_data
        )

    def test_get_item_counts(self):

        prefixspan = PrefixSpanNew()

        sequences = self.prefix_df.get_sequences()

        database = [sequence.sequence_values for sequence in sequences]
        classes = [sequence.class_value for sequence in sequences]

        item, item_neg, item_pos = prefixspan.get_item_counts(database, classes)

        for el in [item, item_neg, item_pos]:
            with self.subTest(msg=f'item: {el}'):
                self.assertIsInstance(el, dict)

    def test_get_frequent_patterns(self):

        prefixspan = PrefixSpanNew(
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


    def test_execute(self):

        prefixspan = PrefixSpanNew()

        patterns = prefixspan.execute(self.raw_data)

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

        self.timestamps = [datetime(2020, 1, 1, 12, i) for i in range(len(self.database))]

        self.id_values = ['a', 'b', 'c', 'd', 'e', 'f']

        # create dataframe entry for each element in database
        records = []

        for i, el in enumerate(self.database):
            for event in el:

                records.append({
                    'id_column': self.id_values[i],
                    'time_column': self.timestamps[i],
                    'event_column': event,
                    'class_column': self.classes[i]
                    })
                
        self.raw_data = pd.DataFrame.from_records(records)

        self.prefix_df = Dataset(
            raw_data=self.raw_data
        )

        self.prefixspan_config = {
            'class_name': 'src.preprocess.extraction.ts_features.PrefixSpanNew',
            'params': {}
        }

    def test__bootstrap(self):

        feat_alg = SPMFeatureSelectorNew(
            prefixspan_config=self.prefixspan_config
        )

        patterns = feat_alg._bootstrap(data=self.raw_data)
    
        for pattern in patterns:
            self.assertIsInstance(pattern, FrequentPatternWithConfidence)

    def test_encode_train(self):

        feat_alg = SPMFeatureSelectorNew(
            prefixspan_config=self.prefixspan_config
        )

        result = feat_alg._encode_train(
            data=self.raw_data
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

        feat_alg = SPMFeatureSelectorNew(
            prefixspan_config=self.prefixspan_config
        )

        kwargs = feat_alg._encode_train(
            data=self.raw_data
        )

        kwargs["data"] = self.raw_data

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