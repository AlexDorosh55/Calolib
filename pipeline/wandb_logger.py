import tempfile
from copy import copy
from typing import Dict, Any, Set, Optional, Iterable, Sequence

import numpy as np
import wandb
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from pipeline.logger import GANLogger, LoggerConfig
from pipeline.run_env import get_wandb_token


class _WandbLogger(GANLogger):
    """
    Should be used only from WandbCM
    """
    WANDB_POINTS_LIMIT = 10_000
    PYPLOT_FORMAT = 'png'
    SUMMARY_METRICS_TABLE_NAME = 'summary_metrics'

    def __init__(self, config: Optional[LoggerConfig] = None, wandb_run=None):
        super().__init__(config=config)
        self.wandb_run = wandb_run
        self.summary_metrics_table = None

    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        # if period not in self._periods:
        wandb.define_metric(period)
        for metric in data:
            wandb.define_metric(metric, step_metric=period)
        logged_dict = copy(data)
        logged_dict[period] = period_index
        wandb.log(logged_dict)

    def log_distribution(self, values: Dict[str, Sequence[float]], name: str,
                         period: str, period_index: int,
                         log_histogram: bool = True) -> None:
        # limit
        if log_histogram:
            vals_per_key_cnt = int(self.WANDB_POINTS_LIMIT / len(values))

            data = []
            for key, vals in values.items():
                if len(vals) > vals_per_key_cnt:
                    vals = np.random.choice(vals, vals_per_key_cnt, replace=False)
                data += [[key, val] for val in vals]
            table = wandb.Table(data=data, columns=['type', 'value'])
            fields = {
                'groupKeys': 'type',
                'value': 'value',
                'title': name,
            }
            composite_histogram = wandb.plot_table(vega_spec_name="trickman/my_histogram",
                                                   data_table=table, fields=fields)
            wandb.log({name: composite_histogram})
        else:
            keys = []
            ys = []

            min_x = min(min(vals) for vals in values.values())
            max_x = max(max(vals) for vals in values.values())

            xs = np.linspace(min_x-0.1, max_x+0.1, num=100)

            for key, vals in values.items():
                keys.append(key)
                kernel = gaussian_kde(vals)
                y = kernel(xs)
                ys.append(y)

            wandb.log({name: wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=f'{name} {period}#{period_index}',
                xname='value')})

    def log_critic_values_distribution(self, critic_values_true: Sequence[float],
                                       critic_values_gen: Sequence[float],
                                       period: str, period_index: int) -> None:
        self.log_distribution(
            values={
                'gen': critic_values_gen,
                'true': critic_values_true,
                },
            name='Critic values',
            period=period, period_index=period_index)

    def log_pyplot(self, name: str, period: str, period_index: int) -> None:
        file = tempfile.NamedTemporaryFile()
        plt.savefig(file, format=self.PYPLOT_FORMAT)
        plt.close()

        wandb.log({name: wandb.Image(file.name)})

        file.close()

    @staticmethod
    def _table_to_dict(wandb_table) -> Dict[str, Any]:
        return dict(zip(wandb_table.columns, wandb_table.data[0]))

    def log_summary_metrics(self, data: Dict[str, Any]) -> None:
        """
        :param data: {metric_name: value}
        """
        prev_data = {}
        if self.summary_metrics_table is not None:
            prev_data = self._table_to_dict(self.summary_metrics_table)
        else:
            artifact_name = f'run-{self.wandb_run.id}-{self.SUMMARY_METRICS_TABLE_NAME}:latest'
            try:
                summary_metrics_table = self.wandb_run.use_artifact(artifact_name).get(self.SUMMARY_METRICS_TABLE_NAME)
                prev_data = self._table_to_dict(summary_metrics_table)
            except wandb.CommError:
                pass

        new_data = prev_data
        new_data.update(data)

        table = wandb.Table(columns=list(new_data.keys()), data=[list(new_data.values())])
        wandb.log({self.SUMMARY_METRICS_TABLE_NAME: table})
        self.summary_metrics_table = table


class WandbCM:
    """
    Wandb logger context manager

    calls wandb.login(), wandb.init() and wandb.finish()

    Use different 'experiment_id's for different runs. Otherwise, the old one will be resumed.
    """
    def __init__(self, project_name: str, experiment_id: str, token: Optional[str] = None,
                 config: Optional[LoggerConfig] = None) -> None:
        """
        :param token: if None, it will be retrieved automatically
        """
        self.project_name = project_name
        self.experiment_id = experiment_id
        if token is None:
            token = get_wandb_token()
        self.token = token
        self.config = config

    @staticmethod
    def _generate_run_id(project_name: str) -> str:
        MODULO = 1_0000_0000
        return str(hash(project_name) % MODULO)

    def __enter__(self) -> _WandbLogger:
        wandb.login(key=self.token)
        run = wandb.init(
            project=self.project_name,
            # id=self._generate_run_id(self.experiment_id),
            name=self.experiment_id,
            # resume='allow',
        )
        return _WandbLogger(config=self.config, wandb_run=run)

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()
