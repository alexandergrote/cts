import hydra
from pathlib import Path
from enum import Enum

CONFIG_FOLDER_NAME = "config"

class Directory:

    ROOT = Path(__file__).parent.parent.parent
    SRC = ROOT / 'src'
    DATA = ROOT / 'data'
    CONFIG = ROOT / CONFIG_FOLDER_NAME
    OUTPUT_DIR = ROOT / "outputs"
    CACHING_DIR = ROOT / "caching"
    MLFLOW_TRACKING_URI = ROOT / "mlruns"
    FIGURES_DIR = ROOT / "figures"

    @classmethod
    def create_dirs(cls):
        dirs = [cls.DATA, cls.CONFIG, cls.OUTPUT_DIR, cls.CACHING_DIR, cls.MLFLOW_TRACKING_URI, cls.FIGURES_DIR]
        for dir in dirs:
            print(dir)
            dir.mkdir(parents=True, exist_ok=True)


class File:
    CONFIG = Directory.CONFIG / "config.yaml"
    MAIN = Directory.SRC / "main.py"


class BaseEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class YamlField(BaseEnum):
    CLASS_NAME = "class_name"
    PARAMS = "params"


class RuleFields(BaseEnum):
    ANTECEDENT = 'antecedent'
    PRECEDEDENT = 'precedent' 
    CONFIDENCE = 'confidence'
    SUPPORT = 'support'
    RANKING = 'ranking'

class EnvMode(BaseEnum):
    DEV = 'dev'
    PROD = 'prod'


HYDRA_CONFIG = {
    'config_path': str(Directory.CONFIG),
    'config_name': File.CONFIG.stem,
    'version_base': "1.1"
}

def get_hydra_output_dir() -> Path:

    try:
        config = hydra.core.hydra_config.HydraConfig
        return Path(config.get().runtime.output_dir)
    
    except ValueError:
        return Directory.OUTPUT_DIR


def replace_placeholder_in_dict(dictionary, placeholder, replacement):
    """
    Replace all occurrences of a placeholder in a dictionary with a replacement.

    Args:
        dictionary (dict): The dictionary to search for placeholders.
        placeholder (str): The placeholder to search for.
        replacement (str): The replacement for the placeholder.

    Returns:
        dict: The dictionary with all placeholders replaced.
    """
    for key, value in dictionary.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    dictionary[key][i] = replace_placeholder_in_dict(
                        dictionary=item,
                        placeholder=placeholder,
                        replacement=replacement
                    )
                elif isinstance(item, str):
                    dictionary[key][i] = item.replace(placeholder, replacement)
        elif isinstance(value, dict):
            dictionary[key] = replace_placeholder_in_dict(
                dictionary=value,
                placeholder=placeholder,
                replacement=replacement
            )
        elif isinstance(value, str):
            dictionary[key] = value.replace(placeholder, replacement)
    return dictionary


Directory.create_dirs()