from abc import abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Set, Optional, Tuple, Union

"""
Currently there's no convenient support for several loggers at one time.
It is expected to be provided if the existing logging interface shows its convenience.
"""


@dataclass
class LoggerConfig:
    ignored_periods: Optional[Set[str]] = None
    ignored_metrics: Optional[Set[str]] = None
    periodic_periods: Optional[Dict[str, int]] = None
    # TODO: добавить возможность отключить логирование training_stats


def get_default_config() -> LoggerConfig:
    return LoggerConfig(
        ignored_periods={'gen_batch', 'disc_batch'}
    )


class GANLogger:
    """An abstract class for GAN loggers"""
    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        self.config = get_default_config() if config is None else config
        self.accumulated_data: Dict[str, Tuple[int, Dict[str, Any]]] = {}   # (period: data with {period: period_index})
        self.training_metrics = defaultdict(dict)  # {period_name: {metric_name: {'min': (min_value, period_index), 'max': (max_value, period_index), 'last': value}}}

    def _update_training_info(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        cur_period_minmax = self.training_metrics[period]
        for metric_name, value in data.items():
            if not isinstance(value, float) and not isinstance(value, int):
                continue
            if metric_name not in cur_period_minmax:
                cur_period_minmax[metric_name] = {'min': (value, period_index), 'max': (value, period_index)}
            else:
                if value < cur_period_minmax[metric_name]['min'][0]:
                    cur_period_minmax[metric_name]['min'] = (value, period_index)
                if value > cur_period_minmax[metric_name]['max'][0]:
                    cur_period_minmax[metric_name]['max'] = (value, period_index)
            cur_period_minmax[metric_name]['last'] = value

    def get_training_metrics(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        {period_name: {metric_name: {'min': (min_value, period_index), 'max': (max_value, period_index)}}}
        """
        return dict(self.training_metrics)

    def log_summary_metrics(self, data: Dict[str, Any]) -> None:
        pass

    def log_running_training_metrics(self, period: str) -> None:
        training_metrics = self.training_metrics[period]
        logged_data = {}
        for metric_name, stats in training_metrics.items():
            for stat_name, value in stats.items():
                full_metric_name = stat_name + ' ' + metric_name
                if stat_name != 'last':
                    value, epoch = value

                if isinstance(value, float):
                    value = round(value, 4)
                if stat_name != 'last':
                    saved_value = f'{value} ({epoch})'
                else:
                    saved_value = value
                logged_data[full_metric_name] = saved_value

        self.log_summary_metrics(logged_data)

    def log_metrics(self, data: Dict[str, Any], period: str, period_index: Optional[int] = None, commit: bool = True) -> None:
        """
        Log values of metrics after some period

        :param data: a dict of metrics values {metric_name: value}
        :param period: the name of a period (e.g., "batch", "epoch")
        :param period_index: the index of a period, if the call is not the first for this period, it may be omitted
        :param commit: if False, data will be accumulated but not logged
        use commit=True only for the last call for the pair (period, period_index)
        """
        if self.config.ignored_periods and period in self.config.ignored_periods:
            return

        if self.config.periodic_periods and period in self.config.periodic_periods and \
           (period_index % self.config.periodic_periods[period]) != 0:
            return

        if self.config.ignored_metrics is not None:
            data = copy(data)
            for metric in copy(data):
                if metric in self.config.ignored_metrics:
                    data.pop(metric)

        data = copy(data)
        if period in self.accumulated_data:
            prev_period_index, prev_data = self.accumulated_data[period]
            if period_index is not None and prev_period_index != period_index:
                raise RuntimeError(f'Trying to log data for the {period} #{period_index} while the data for the {period} #{prev_period_index} was not logged')
            period_index = prev_period_index
            data.update(prev_data)

        assert period_index is not None, 'Period index is not specified'

        self._update_training_info(data, period=period, period_index=period_index)

        self.accumulated_data[period] = (period_index, data)

        if commit:
            self.log_running_training_metrics(period=period)
            self._log_metrics(data, period, period_index)
            self.accumulated_data.pop(period)

    def commit(self, period: str):
        if period in self.accumulated_data:
            self.log_metrics(data={}, period=period, commit=True)

    @abstractmethod
    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        pass

    # Optional for implementation
    # it may plot the histogram, density approximation, etc.
    # or print some statistics of the distribution (mean, variance, etc.)
    def log_distribution(self, values: Dict[str, Iterable[float]], name: str,
                         period: str, period_index: int) -> None:
        pass

    # Optional for implementation
    def log_critic_values_distribution(self, critic_values_true: Iterable[float],
                                       critic_values_gen: Iterable[float],
                                       period: str, period_index: int) -> None:
        """
        Log the distributions of critic values

        :param critic_values_gen: critic values for the generated data
        :param critic_values_true: critic values for the true data, must be the same length as
        `critic_values_gen`
        """
        pass

    # Optional for implementation
    def log_pyplot(self, name: str, period: str, period_index: int) -> None:
        """
        Log the current opened figure.
        A matplotlib.pyplot figure is expected to be created before a call.
        """
        pass
