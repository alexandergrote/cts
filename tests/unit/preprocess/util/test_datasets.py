import unittest
import pandera
import pandas as pd


from src.preprocess.util.types import AnnotatedSequence
from src.preprocess.util.datasets import Dataset, DatasetUniqueRules, DatasetUniqueRulesSchema

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


class TestDatasetUniqueRules(unittest.TestCase):

    def setUp(self) -> None:

        self.data = pd.DataFrame({
            DatasetUniqueRulesSchema.id_column: ['a', 'b'],
            DatasetUniqueRulesSchema.delta_confidence: [[0.1], [0.3, 0.4]],
            DatasetUniqueRulesSchema.inverse_entropy: [[0.5], [-0.7, 0.8]],
            DatasetUniqueRulesSchema.support: [[1/3], [3/7, 4/7]],
        })

    def test_unique_rules(self):

        with self.assertRaises(pandera.errors.SchemaError):

            data = self.data.copy(deep=True)
            data[DatasetUniqueRulesSchema.id_column] = 'a'

            DatasetUniqueRulesSchema.validate(data)


    def test_ranked_rules(self):

        dataset = DatasetUniqueRules(data=self.data)

        ranked_rules = dataset.rank_rules(
            criterion=DatasetUniqueRulesSchema.inverse_entropy,
        )

        expected = [('a', 0.5), ('b', 0.05)]

        for i, el in enumerate(ranked_rules):
            self.assertEqual(el[0], expected[i][0])
            self.assertEqual(round(el[1], 5), expected[i][1])


    def test_weighted_ranked_rules(self):

        dataset = DatasetUniqueRules(data=self.data)

        ranked_rules = dataset.rank_rules(
            criterion=DatasetUniqueRulesSchema.delta_confidence,
            weighted_by_support=True,
        )

        expected = [
            ('b', ((0.3*3/7) + (0.4*4/7))/2),
            ('a', 0.1*1/3),            
        ]

        for i, el in enumerate(ranked_rules):
            self.assertEqual(el[0], expected[i][0])
            self.assertEqual(round(el[1], 5), round(expected[i][1], 5))