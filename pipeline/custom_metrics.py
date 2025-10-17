import torch

from pipeline.metrics import *
from pipeline.physical_metrics.calogan_prd import get_energy_embedding


def create_prd_energy_embed(num_clusters: int = 20, num_runs: int = 10, enforce_balance: bool = True):
    calculated_metric = PhysicsPRDMetric(num_clusters=num_clusters, num_runs=num_runs,
                                         enforce_balance=enforce_balance)
    calculated_metric.NAME = 'Energy ' + calculated_metric.NAME

    metric = TransformData(
        calculated_metric,
        transform_fn=get_energy_embedding,
    )
    return metric


def create_conditional_prd_energy_embed(num_clusters: int = 20, num_runs: int = 10, enforce_balance: bool = True):
    calculated_metric = AveragePRDAUCMetric(num_clusters=num_clusters, num_runs=num_runs,
                                            enforce_balance=enforce_balance)
    calculated_metric.NAME = 'Energy embed. ' + calculated_metric.NAME

    metric = TransformData(
        ConditionBinsMetric(
            calculated_metric,
            dim_bins=torch.Tensor([3, 3]),
            condition_index=0
        ),
        transform_fn=get_energy_embedding,
    )
    return metric


def create_prd_physics_statistics(num_clusters: int = 20, num_runs: int = 10, enforce_balance: bool = True):
    calculated_metric = PhysicsPRDMetric(num_clusters=num_clusters, num_runs=num_runs,
                                         enforce_balance=enforce_balance)
    calculated_metric.NAME = 'PhysStats ' + calculated_metric.NAME

    metric = TransformData(
        calculated_metric,
        transform_fn=DataStatisticsCombiner(
            *[statistic.evaluate_statistic for statistic in PHYS_STATISTICS]
        )
    )
    return metric


def create_conditional_prd_physics_statistics(num_clusters: int = 20, num_runs: int = 10, enforce_balance: bool = True):
    calculated_metric = AveragePRDAUCMetric(num_clusters=num_clusters, num_runs=num_runs,
                                            enforce_balance=enforce_balance)
    calculated_metric.NAME = 'PhysStats ' + calculated_metric.NAME

    metric = TransformData(
        ConditionBinsMetric(
            calculated_metric,
            dim_bins=torch.Tensor([3, 3]),
            condition_index=0
        ),
        transform_fn=DataStatisticsCombiner(
            *[statistic.evaluate_statistic for statistic in PHYS_STATISTICS]
        )
    )
    return metric


__all__ = ['create_prd_energy_embed', 'create_conditional_prd_energy_embed',
           'create_prd_physics_statistics', 'create_conditional_prd_physics_statistics']
