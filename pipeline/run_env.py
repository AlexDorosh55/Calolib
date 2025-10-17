"""
Something specific to the environment in which the pipeline is running
"""
import os
import sys
from enum import Enum, auto


class Environment(Enum):
    LOCAL = auto()
    KAGGLE = auto()


# if this env var is not defined, the environment is determined automatically
ENV_VAR_NAME = 'GAN_TRAINER_ENV'
ENVS_BY_NAME = {
    'LOCAL': Environment.LOCAL,
    'KAGGLE': Environment.KAGGLE,
}


def get_local_env() -> Environment:
    if ENV_VAR_NAME in os.environ:
        env_name = os.environ[ENV_VAR_NAME]
        env = ENVS_BY_NAME.get(env_name, None)
        if env is not None:
            return env
        else:
            print(f'The env var {ENV_VAR_NAME} is defined but has incorrect value.'
                  f'Trying to determine the environment automatically',
                  file=sys.stderr)

    if 'KAGGLE_URL_BASE' in os.environ:
        return Environment.KAGGLE
    else:
        return Environment.LOCAL


ENV = get_local_env()


def get_wandb_token() -> str:
    if ENV is Environment.LOCAL:
        return os.getenv('WANDB_TOKEN')
    elif ENV is Environment.KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('WANDB_TOKEN')
