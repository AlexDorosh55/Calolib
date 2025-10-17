"""
Auxiliary functions to easily visualize the results of GANs applied to the Physics task
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualization_aux.common import imshow


def energy_imshow(energy, ax=None, cmap='inferno', log_transform: bool = True):
    if log_transform:
        energy = torch.log1p(energy)
    imshow(energy, ax=ax, cmap=cmap, affine=False)


def add_noise(arr):
    noise_coefs = 1 + np.random.normal(0, 0.1, size=arr.shape)
    return arr * noise_coefs


def get_test_data(global_config):
    """
    Примеры данных для визуалиации (отбирались вручную)
    """
    data_train = np.load(os.path.join(global_config.paths.data_dir_path,
                                      'caloGAN_case11_5D_120K.npz'))
    samples_indices = [0, 31516, 62946, 37323, 57956]

    pure_points = torch.Tensor(
        data_train['ParticlePoint'][samples_indices, :2]
    )
    pure_momentums = torch.Tensor(
        data_train['ParticleMomentum'][samples_indices]
    )

    noised_points = torch.Tensor(add_noise(
        data_train['ParticlePoint'][samples_indices, :2]
    ))
    noised_momentums = torch.Tensor(add_noise(
        data_train['ParticleMomentum'][samples_indices]
    ))

    energy = data_train['EnergyDeposit'][samples_indices]
    points = torch.vstack([pure_points, noised_points])
    momentums = torch.vstack([pure_momentums, noised_momentums])

    return energy, points, momentums
