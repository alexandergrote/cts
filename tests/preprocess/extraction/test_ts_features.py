import pandas as pd
import unittest

from datetime import datetime

from src.preprocess.extraction.ts_features import PrefixSpanDataset, AnnotatedSequence, PrefixSpanNew


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

        dataset = PrefixSpanDataset(
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


    def test_get_item_counts(self):

        prefixspan = PrefixSpanNew()

        prefix_df = PrefixSpanDataset(
            raw_data=self.raw_data
        )

        sequences = prefix_df.get_sequences()

        database = [sequence.sequence_values for sequence in sequences]
        classes = [sequence.class_value for sequence in sequences]

        item, item_neg, item_pos = prefixspan.get_item_counts(database, classes)

        for el in [item, item_neg, item_pos]:
            with self.subTest(msg=f'item: {el}'):
                self.assertIsInstance(el, dict)

    def test_execute(self):

        prefixspan = PrefixSpanNew()

        result = prefixspan.execute(self.raw_data)

        print(result)


if __name__ == '__main__':
    unittest.main()