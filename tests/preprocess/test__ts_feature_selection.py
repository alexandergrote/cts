import unittest
from src.preprocess.ts_feature_selection import PrefixSpan

class TestPrefixSpan(unittest.TestCase):

    def setUp(self):
        self.prefix_span = PrefixSpan(
            id_columns=['id'],
            event_column='event',
        )

    def test__get_support_dict(self):
        
        sequences = [
            ['a', 'b', 'c'],
            ['a', 'b', 'd'],
        ]

        support_dict = self.prefix_span._get_support_dict(sequences)

        self.assertEqual(support_dict['a'], 2)
        self.assertEqual(support_dict['a --> c'], 1)

        with self.assertRaises(KeyError):
            support_dict['c --> a']
