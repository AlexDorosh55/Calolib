from abc import abstractmethod
from typing import Callable

import torch


class Regularizer:
    """
    Regularization loss
    """
    def step(self) -> None:
        """
        Call after each epoch
        """
        pass

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass


class BasicRegularizer(Regularizer):
    def __init__(self, func: Callable[[], torch.Tensor]) -> None:
        self._func = func

    def __call__(self) -> torch.Tensor:
        return self._func()


class NoPenaltyIntervalRegularizer(Regularizer):
    def __init__(self, regularizer: Regularizer, lim: float):
        self.inner_regularizer = regularizer
        self.lim = lim

    def __call__(self) -> torch.Tensor:
        val = self.inner_regularizer()
        if val < self.lim:
            return torch.tensor(0.)
        else:
            return val - self.lim


class PowRegularizer(Regularizer):
    def __init__(self, regularizer: Regularizer, pow: float):
        self.inner_regularizer = regularizer
        self.pow = pow

    def __call__(self) -> torch.Tensor:
        return self.inner_regularizer() ** self.pow


class MultiplierRegularizer(Regularizer):
    def __init__(self, regularizer: Regularizer, start_value: float, add_value: float = 0):
        """
        :param start_value: initial multiplier
        :param add_value: value that is added to multiplier after each step(), if None
        """
        self.inner_regularizer = regularizer
        self.multiplier = start_value
        self.add_value = add_value

    def step(self) -> None:
        self.multiplier += self.add_value

    def __call__(self) -> torch.Tensor:
        return self.multiplier * self.inner_regularizer()


__all__ = ['Regularizer', 'BasicRegularizer', 'PowRegularizer', 'MultiplierRegularizer']
