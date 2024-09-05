import unittest
from typing import List
from src.util.load_hydra import get_hydra_config
from src.main import main


class TestSynthetic(unittest.TestCase):

    def _integration_test(self, overrides: List[str] = []):

        default_overrides = [
            'fetch_data=synthetic', 
            'fetch_data.params.n_samples=1000',
            'export=dummy'
        ]

        final_overrides = overrides + default_overrides

        cfg = get_hydra_config(overrides=final_overrides)

        self.assertIsNone(main(cfg))

    def test_preprocess(self):

        selection_options = ['self', 'rf', 'mutinfo', 'mrmr']
        encoding_options = ['spm', 'oh']

        selection_options = ['self']
        encoding_options = ['oh']

        for selection in selection_options:
            for encoding in encoding_options:

                statement = f"preprocess={selection}_{encoding}"

                with self.subTest(msg=statement):
                    self._integration_test([
                        f"preprocess={selection}_{encoding}",
                    ])
        


        


if __name__ == '__main__':
    unittest.main()
