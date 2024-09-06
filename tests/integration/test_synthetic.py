import unittest
from unittest.mock import patch
from typing import List
from pathlib import Path
from tempfile import TemporaryDirectory
from src.util.load_hydra import get_hydra_config
from src.main import main


temp_dir = TemporaryDirectory()

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

    @patch("src.main.get_hydra_output_dir", return_value=Path(temp_dir.name))
    def test_preprocess(self, mock_output_dir):

        selection_options = ['self', 'rf', 'mutinfo', 'mrmr']
        encoding_options = ['spm', 'oh']
        n_features = ['null', 1]

        for selection in selection_options:
            for encoding in encoding_options:
                for n_feature in n_features:

                    statement = f"preprocess={selection}_{encoding}_{str(n_feature)}"

                    with self.subTest(msg=statement):
                        self._integration_test([
                            f"preprocess={selection}_{encoding}",
                            f"preprocess.params.selector.params.n_features={n_feature}"
                        ])
        


        


if __name__ == '__main__':
    unittest.main()
