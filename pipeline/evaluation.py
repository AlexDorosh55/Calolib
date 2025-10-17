from typing import Dict, Optional, Tuple

import torch.utils.data

from pipeline.gan import GAN
from pipeline.logger import GANLogger
from pipeline.metrics import Metric, unravel_metric_results
from pipeline.results_storage import ResultsStorage


def evaluate_model(gan_model: GAN, val_dataset: torch.utils.data.Dataset,
                   metric: Metric, logger: GANLogger) -> None:
    metric_results = metric.evaluate(gan_model=gan_model, val_dataset=val_dataset)

    new_data = unravel_metric_results(metric, metric_results)
    logger.log_summary_metrics(new_data)

# def save_training_stats(model_name: str, storage: ResultsStorage, epochs_training_stats: Dict[str, Dict[str, Tuple]]) -> None:
#     """
#     :param epochs_training_stats: {metric_name: {'min': (min_value, period_index), 'max': (max_value, period_index)}}
#     """
#     exp_info = storage.get_experiment_info(model_name)
#     exp_result = exp_info.get_result()
#
#     for metric_name, stats in epochs_training_stats.values():
#         for stat_name, (value, epoch) in stats:
#             full_metric_name = stat_name + ' ' + metric_name
#             if isinstance(value, float):
#                 value = round(value, 4)
#             saved_value = f'{value} ({epoch})'
#             exp_result.add_metric(full_metric_name, saved_value)
#
#
# def evaluate_model(model_name: str, gan_model: GAN, val_dataset: torch.utils.data.Dataset,
#                    metric: Metric, storage: ResultsStorage, force_rewrite: bool = False) -> None:
#     exp_info = storage.get_experiment_info(model_name)
#     exp_result = exp_info.get_result()
#
#     metric_results = metric.evaluate(gan_model=gan_model, val_dataset=val_dataset)
#
#     old_metrics = exp_result.metrics
#     for metric_name, res in unravel_metric_results(metric, metric_results):
#         if metric_name in old_metrics and not force_rewrite:
#             continue
#         exp_result.add_metric(metric_name, res)
