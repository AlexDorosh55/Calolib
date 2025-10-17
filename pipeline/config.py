import os
from dataclasses import dataclass, is_dataclass
from typing import Dict

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


CONFIGS_DIR = './config'

"""
Used config-files:
paths.yaml
logger.yaml
"""


def _load_yaml(filename: str) -> Dict:
    filepath = os.path.join(CONFIGS_DIR, filename)
    with open(filepath, 'r') as file:
        data = yaml.load(file, Loader=Loader)
    return data


def _load_config(cls, data: Dict):
    args = {}
    for arg_name, value in data.items():
        field_type = cls.__annotations__.get(arg_name, None)
        if is_dataclass(field_type):
            args[arg_name] = _load_config(field_type, value)
        else:
            args[arg_name] = value
    return cls(**args)


def load_config(config_cls, filename: str):
    try:
        data = _load_yaml(filename)
        return _load_config(config_cls, data)
    except FileNotFoundError:
        return None


@dataclass
class ExperimentsConfig:
    """
    Attributes:
        experiments_dir: a directory where experiments data (checkpoints, models, configs) will be stored
    """
    experiments_dir: str = './experiments'


@dataclass
class PathsConfig:
    """
    Attributes:
        data_dir_path: a directory with data
    """
    data_dir_path: str
    experiments: ExperimentsConfig = ExperimentsConfig()


@dataclass
class LoggerConfig:
    project_name: str
    enable_logging: bool = True


@dataclass
class GlobalConfig:
    paths: PathsConfig
    logger: LoggerConfig


def load_global_config() -> GlobalConfig:
    paths_config = load_config(PathsConfig, 'paths.yaml')
    logger_config = load_config(LoggerConfig, 'logger.yaml')
    return GlobalConfig(paths=paths_config, logger=logger_config)
