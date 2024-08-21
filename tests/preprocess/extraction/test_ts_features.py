import pandas as pd

from unittest import TestCase

from src.preprocess.extraction.ts_features import PrefixSpanDataset, Sequence, PrefixSpan


class TestPrefixSpanDataset(TestCase):

    def setUp(self) -> None:

        self.id_columns = ['id1', 'id2']
        self.event_column = 'event'

        self.raw_data = pd.DataFrame({
            'id1': ['a', 'a', 'a', 'b', 'b'],
            'id2': ['aa', 'aa', 'a', 'bb', 'bb'],
            'event': [1,2,3,5,4]
        })

    def test_get_sequences(self):

        dataset = PrefixSpanDataset(
            id_columns=self.id_columns,
            event_column=self.event_column,
            raw_data=self.raw_data
        )

        sequences = dataset.get_sequences()

        self.assertIsInstance(sequences, list)
        self.assertTrue(all([isinstance(el, Sequence) for el in sequences]))
        self.assertEqual([el.sequence_values for el in sequences], [['1','2'], ['5','4']])
        self.assertEqual([el.id_values for el in sequences], [['a', 'aa'], ['b', 'bb']])


class TestPrefixSpan(TestCase):

    def setUp(self) -> None:
        
        self.sequences = [
            Sequence(id_values=['a', 'aa'], sequence_values=['1', '2']),
            Sequence(id_values=['b', 'bb'], sequence_values=['1', '3', '2'])
        ]

        self.prefix_span = PrefixSpan()

    def test_get_combinations(self):

        combinations = self.prefix_span.get_combinations(self.sequences[1].sequence_values)

        combinations_expected = [('1', '3', '2'), ('1', '3'), ('1', '2'), ('3', '2'), ('1',), ('2',), ('3',)]

        # check if all expected combinations are contained
        for comb_expected in combinations_expected:
            self.assertIn(comb_expected, combinations)

        # check if there are additional combinations
        self.assertEqual(len(combinations), len(combinations_expected))

    
    def test_get_support(self):

        support = self.prefix_span.get_support(self.sequences)

        expected_support = {
            '1 --> 2': 2, 
            '2': 2, 
            '1': 2, 
            '3 --> 2': 1, 
            '1 --> 3 --> 2': 1, 
            '1 --> 3': 1, 
            '3': 1
        }

        self.assertEqual(support, expected_support)
        self.assertTrue('2 --> 1' not in support)


