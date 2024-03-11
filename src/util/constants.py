from pathlib import Path
from enum import Enum


class Directory:

    ROOT = Path(__file__).parent.parent.parent
    SRC = ROOT / 'src'
    DATA = ROOT / 'data'
    CONFIG = ROOT / "config"
    OUTPUT_DIR = ROOT / "outputs"
    CACHING_DIR = ROOT / "caching"
    MLFLOW_TRACKING_URI = ROOT / "mlruns"


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
