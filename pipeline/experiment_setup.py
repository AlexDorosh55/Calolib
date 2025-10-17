"""
Local config which is the same for all experiments running in the environment
"""
import contextlib
from typing import Optional

from pipeline import logger
from pipeline.config import load_global_config
from pipeline.logger import LoggerConfig
from pipeline.storage import ExperimentsStorage
from pipeline.wandb_logger import WandbCM


global_config = load_global_config()


def init_storage() -> ExperimentsStorage:
    # === config variables ===
    experiments_dir = global_config.paths.experiments.experiments_dir
    checkpoint_filename = './training_checkpoint'
    model_state_filename = './model_state'
    # ========================
    return ExperimentsStorage(experiments_dir=experiments_dir, checkpoint_filename=checkpoint_filename,
                              model_state_filename=model_state_filename)


experiments_storage = init_storage()


def init_logger(model_name: str = '', config: Optional[LoggerConfig] = None):
    if not global_config.logger.enable_logging:
        return None
    project_name = global_config.logger.project_name
    config = config or logger.get_default_config()
    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(project_name=project_name, experiment_id=model_name, config=config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm
