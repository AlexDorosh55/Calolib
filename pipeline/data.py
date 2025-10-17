from itertools import cycle
from typing import Tuple, List, Optional, Union, Any, Sequence

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms


"""
Датасеты могут быть двух типов:
1. Элемент - число или тензор. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
y - либо число, либо тензор, либо tuple с числами/тензорами

Обёртка в виде UnifiedDatasetWrapper приводит оба типа датасетов ко 2-му (для датасета 1-го типа y = None).
Обёртку следует использовать пользователю.
"""


class RandomDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size,
                                                               drop_last=False)

        super().__init__(dataset, batch_sampler=random_sampler, *args, **kwargs)


def get_random_infinite_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
    return cycle(RandomDataloader(dataset, batch_size=batch_size, *args, **kwargs))


def collate_fn(els_list: Sequence[Union[Tuple, int, torch.Tensor]]):
    if isinstance(els_list[0], tuple):
        return tuple(collate_fn(list(a)) for a in zip(*els_list))
    elif isinstance(els_list[0], int):
        return torch.Tensor(els_list)
    elif isinstance(els_list[0], torch.Tensor):
        return torch.stack(tuple(els_list))
    elif els_list[0] is None:
        return None
    else:
        raise RuntimeError


def stack_batches(batches_list):
    if isinstance(batches_list[0], tuple):
        return tuple(stack_batches(list(a)) for a in zip(*batches_list))
    elif isinstance(batches_list[0], torch.Tensor):
        return torch.concat(batches_list, dim=0)
    elif batches_list[0] is None:
        return None


def move_batch_to(batch, device):
    if isinstance(batch, tuple):
        return tuple(move_batch_to(subbatch, device) for subbatch in batch)
    elif batch is None:
        return None
    else:
        return batch.to(device)


class LinearTransform(nn.Module):
    def __init__(self, min_to: float, max_to: float, dim: int) -> None:
        super().__init__()
        self.min_to = min_to
        self.max_to = max_to
        self.dim = dim

    def forward(self, X) -> torch.Tensor:
        mins = X.min(dim=self.dim).values
        maxs = X.max(dim=self.dim).values

        coefs = (self.max_to - self.min_to) / (maxs - mins)
        biases = self.min_to - coefs * mins

        shape = list(X.shape)
        shape[self.dim] = 1

        coefs = coefs.reshape(*shape)
        biases = biases.reshape(*shape)

        y = coefs * X + biases
        y = torch.clip(y, self.min_to, self.max_to)  # clipping for making sure

        return y


class ExtractIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Union[Tuple[int], int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n: int):
        obj = self.dataset[n]
        if isinstance(self.indices, int):
            return obj[self.indices]
        else:
            return tuple(obj[i] for i in self.indices)


class UnifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    Обёртка для поддержки датасетов обоих типов
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.inverse_transform = getattr(dataset, 'inverse_transform', None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, n: int) -> Tuple[Any, Any]:
        element = self.dataset[n]
        if isinstance(element, tuple):
            assert len(element) == 2
            x, y = element
        else:
            x, y = element, None
        return x, y


def get_default_image_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # Переводим цвета пикселей в отрезок [-1, 1] афинным преобразованием, изначально они в отрезке [0, 1]
        transforms.Normalize(tuple(0.5 for _ in range(dim)), tuple(0.5 for _ in range(dim)))
    ])


default_image_transform = get_default_image_transform(3)


def get_cifar_10_dataset(root='./cifar10', train: bool = True, keep_labels: bool = True, kept_labels: Optional[List[int]] = None):
    """
    :param kept_labels: images with which labels should be used
    """
    cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                                 transform=default_image_transform, )

    if kept_labels is not None:
        kept_indices = []
        for i in range(len(cifar_dataset)):
            if cifar_dataset[i][1] in kept_labels:
                kept_indices.append(i)

        cifar_dataset = torch.utils.data.Subset(cifar_dataset, kept_indices)

    if not keep_labels:
        cifar_dataset = ExtractIndicesDataset(cifar_dataset, indices=0)
    return cifar_dataset


def get_mnist_dataset(root='./mnist', train: bool = True, keep_labels: bool = True):
    mnist_dataset = torchvision.datasets.MNIST(root=root, train=train, download=True,
                                               transform=get_default_image_transform(1))

    if not keep_labels:
        mnist_dataset = ExtractIndicesDataset(mnist_dataset, indices=0)
    return mnist_dataset


class PhysicsDataset(torch.utils.data.Dataset):
    """
    one element: (energy deposit, (point, momentum))
    """
    def __init__(self, energy: torch.Tensor, point: torch.Tensor, momentum: torch.Tensor,
                 transform=None, inverse_transform=None) -> None:
        """
        TODO: указать сигнатуры transform и inverse_transform через typing
        transform(energy, point, momentum) - tensors as batches or single
        inverse_transform(energy, point, momentum)
        """
        self.transform = transform
        self.inverse_transform = inverse_transform  # for outer use

        if transform is not None:
            energy = self.transform(energy)

        self.energy = energy
        self.point = point
        self.momentum = momentum

    def __getitem__(self, idx: int) -> tuple:
        return self.energy[idx], (self.point[idx], self.momentum[idx])

    def __len__(self) -> int:
        return self.energy.shape[0]


# принимают batch-и x-ов
def log1p_transform(x: torch.Tensor):
    return torch.log1p(x)


def log1p_inverse_transform(x: torch.Tensor):
    return torch.expm1(x)


def get_physics_dataset(path: str, train: bool = True, val_ratio: float = 0.5,
                        log1p_energy: bool = True) -> torch.utils.data.Dataset:
    TRAIN_VAL_SPLIT_SEED = 0x3df3fa

    data_train = np.load(path)

    np.random.seed(TRAIN_VAL_SPLIT_SEED)
    dataset_size = len(data_train['EnergyDeposit'])
    val_size = int(dataset_size * val_ratio)

    all_indices = np.arange(dataset_size)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    val_mask = np.zeros(dataset_size, dtype=bool)
    val_mask[val_indices] = True
    train_indices = all_indices[~val_mask]
    indices = train_indices if train else val_indices

    energy = torch.tensor(data_train['EnergyDeposit'][indices]).float()
    energy = torch.permute(energy, dims=(0, 3, 1, 2))
    point = torch.tensor(data_train['ParticlePoint'][:, :2][indices]).float()
    momentum = torch.tensor(data_train['ParticleMomentum'][indices]).float()

    transform, inverse_transform = None, None
    if log1p_energy:
        transform = log1p_transform
        inverse_transform = log1p_inverse_transform

    return PhysicsDataset(energy, point, momentum,
                          transform=transform, inverse_transform=inverse_transform)
