import unittest
import pandera
import pandas as pd

from src.preprocess.util.datasets import DatasetUniqueRules, DatasetUniqueRulesSchema


class TestDatasetUniqueRules(unittest.TestCase):

    def setUp(self) -> None:

        self.data = pd.DataFrame({
            DatasetUniqueRulesSchema.id_column: ['a', 'b'],
            DatasetUniqueRulesSchema.delta_confidence: [[0.1], [0.3, 0.4]],
            DatasetUniqueRulesSchema.centered_inverse_entropy: [[0.5], [-0.7, 0.8]],
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
            criterion=DatasetUniqueRulesSchema.centered_inverse_entropy,
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