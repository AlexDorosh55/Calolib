"""
Предикаты были созданы для удобного контроля эпох, в которые вычисляются метрики, во время обучения.
Главный класс TrainPredicate представляет собой просто обёртку над функцией-предикатом.
"""
from abc import abstractmethod


class TrainPredicate:
    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        pass

    def __and__(self, pred2: 'TrainPredicate') -> 'TrainPredicate':
        return AndPredicate(self, pred2)


# sample predicates
class IgnoreFirstNEpochsPredicate(TrainPredicate):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, epoch: int, *args, **kwargs) -> bool:
        return epoch > self.n


class EachNthEpochPredicate(TrainPredicate):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, epoch: int, *args, **kwargs) -> bool:
        return epoch % self.n == 0


class AndPredicate(TrainPredicate):
    def __init__(self, pred1: TrainPredicate, pred2: TrainPredicate):
        self.pred1 = pred1
        self.pred2 = pred2

    def __call__(self, *args, **kwargs) -> bool:
        return self.pred1(*args, **kwargs) and self.pred2(*args, **kwargs)
