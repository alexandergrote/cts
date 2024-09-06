import pandas as pd
import unittest

from src.preprocess.util.types import BootstrapRound 
from src.preprocess.util.datasets import Dataset, DatasetSchema
from src.preprocess.extraction.ts_features import SPMFeatureSelector


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

        bootstrap_rounds = feat_alg._bootstrap(data=self.prefix_df.raw_data)
    
        for round in bootstrap_rounds:
            self.assertIsInstance(round, BootstrapRound)

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

        # check if class column is present
        self.assertTrue(DatasetSchema.class_column in result_df.columns)

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


if __name__ == '__main__':
    unittest.main()