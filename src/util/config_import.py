from hydra import compose, initialize
from pydantic import BaseModel
from typing import Optional, List
from omegaconf import OmegaConf
from pathlib import Path

from src.util.constants import HYDRA_CONFIG, CONFIG_FOLDER_NAME

class YamlConfigLoader(BaseModel):
    
    @staticmethod
    def read_yaml(key: Optional[str] = None, overrides: Optional[List[str]] = None) -> OmegaConf:

        config_name = HYDRA_CONFIG.pop('config_name')

        config_path = Path(f'../../{CONFIG_FOLDER_NAME}')  # relative path to config needed
        HYDRA_CONFIG['config_path'] = str(config_path)

        with initialize(**HYDRA_CONFIG):
            cfg = compose(config_name=config_name, overrides=overrides)

        if key is not None:
            return cfg[key]

        return cfg
    
if __name__ == '__main__':

    config = YamlConfigLoader.read_yaml(key='fetch_data', overrides=['fetch_data=malware'])

    print(config)