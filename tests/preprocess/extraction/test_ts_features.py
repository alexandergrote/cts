import pandas as pd

from unittest import TestCase

from src.preprocess.extraction.ts_features import PrefixSpanDataset, Sequence


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

