"""Знает об устройстве директорий и именах файлов"""
import os
import shutil
from enum import Enum
from typing import Optional, Dict

import torch

from pipeline.device import get_local_device
from pipeline.config import CONFIGS_DIR


class ModelParts(Enum):
    CONFIG = 'config'
    TRAINING_CHECKPOINT = 'training_checkpoint'
    TRAINED_MODEL = 'trained_model'


class ModelDir:
    def __init__(self, model_dirpath: str,
                 config_dirname: str = 'config',
                 checkpoint_filename: str = 'training_checkpoint',
                 model_state_filename: str = 'model_state') -> None:
        self.model_dirpath = model_dirpath
        self.parts_filepaths = {
            ModelParts.CONFIG: config_dirname,
            ModelParts.TRAINING_CHECKPOINT: checkpoint_filename,
            ModelParts.TRAINED_MODEL: model_state_filename,
        }

        config_dirpath = self.get_part_filepath(ModelParts.CONFIG)
        if not os.path.exists(config_dirpath):
            os.mkdir(config_dirpath)

    def get_part_filepath(self, part: ModelParts):
        """
        Can be either path to a file or path to a directory
        """
        return os.path.join(self.model_dirpath, self.parts_filepaths[part])

    def get_checkpoint_state(self) -> Optional[dict]:
        filepath = self.get_part_filepath(ModelParts.TRAINING_CHECKPOINT)
        if not os.path.exists(filepath):
            return None
        checkpoint = torch.load(filepath, map_location=get_local_device())
        return checkpoint

    def save_checkpoint_state(self, checkpoint: dict) -> None:
        torch.save(checkpoint, self.get_part_filepath(ModelParts.TRAINING_CHECKPOINT))


class ExperimentsStorage:
    SAVED_CONFIG_FILES = ['run_experiment.py']

    def __init__(self, experiments_dir: str = './experiments', **model_dir_kwargs) -> None:
        self.experiments_dir = experiments_dir
        self.model_dir_kwargs = model_dir_kwargs
        if not os.path.exists(self.experiments_dir):
            os.mkdir(self.experiments_dir)

    def get_model_dirpath(self, model_name: str) -> str:
        return os.path.join(self.experiments_dir, model_name)

    def get_model_dir(self, model_name: str) -> ModelDir:
        model_dirpath = self.get_model_dirpath(model_name)
        if not os.path.exists(model_dirpath):
            os.mkdir(model_dirpath)
        return ModelDir(model_dirpath=model_dirpath, **self.model_dir_kwargs)

    def load_config(self, model_name: str) -> None:
        """
        Loads config for the model as the current
        """
        model_dir = self.get_model_dir(model_name)
        config_dirpath = model_dir.get_part_filepath(ModelParts.CONFIG)

        for config_file in self.SAVED_CONFIG_FILES:
            from_filepath = os.path.join(config_dirpath, config_file)
            to_filepath = os.path.join(CONFIGS_DIR, config_file)
            shutil.copy(from_filepath, to_filepath)
        print(f'Config for "{model_name}" was loaded')

    def save_config(self, model_name: str) -> None:
        """
        Saves the current config for the model
        """
        model_dir = self.get_model_dir(model_name)
        to_dirpath = model_dir.get_part_filepath(ModelParts.CONFIG)
        if not os.path.exists(to_dirpath):
            os.mkdir(to_dirpath)
        for config_file in self.SAVED_CONFIG_FILES:
            from_filepath = os.path.join(CONFIGS_DIR, config_file)
            to_filepath = os.path.join(to_dirpath, config_file)
            shutil.copy(from_filepath, to_filepath)
        print(f'Config for "{model_name}" was saved')
