import math
from abc import abstractmethod
from typing import TypeVar, Type, Optional, List, Iterable, FrozenSet, Callable

import torch
from torch import nn


"""
тут можно ещё подумать над удобной реализацией
"""


class ModuleNotSupported(Exception):
    pass


class Normalizer(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        # TODO: кидаем исключение ModuleNotSupported, если не поддерживаем модуль
        self.module = module

    @abstractmethod
    def update_stats(self, **kwargs) -> None:
        """
        Предполагаем, что эта функция вызывается после обновления весов
        """
        pass

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(X, *args, **kwargs)

    def train(self, mode: bool = True) -> 'Normalizer':
        self.module.train(mode)
        return self

    def eval(self) -> 'Normalizer':
        self.module.eval()
        return self


T = TypeVar('T', bound=nn.Module)


class ClippingNormalizer(Normalizer):
    def __init__(self, module: T, clip_c: float) -> None:
        """
        Clip in range [-clip_c, clip_c]
        """
        super().__init__(module)
        self.register_buffer('clip_c', torch.tensor(clip_c))

        if type(module) not in [nn.Linear]:
            raise ModuleNotSupported

    def update_stats(self, **kwargs) -> None:
        with torch.no_grad():
            for param in self.module.parameters():
                param.clip_(min=-self.clip_c, max=self.clip_c)


# class SpectralNormApproximator:
#     """
#     Класс, вычисляющий оценку спектральной нормы для слоя
#     """
#     def __init__(self, module: T) -> None:
#         if isinstance(module, nn.Linear):
#             self.weight_matrix_fn = lambda: module.weight.data
#         elif isinstance(module, nn.Conv2d):
#             self.weight_matrix_fn = lambda: module.weight.data.reshape(
#                 (module.weight.data.shape[0], -1))
#         else:
#             raise ModuleNotSupported
#
#         self.weight_matrix_shape = self.weight_matrix_fn().shape
#         self.u = nn.Parameter(2*torch.rand(self.weight_matrix_shape[0], 1, requires_grad=False)-1)
#         self.v = nn.Parameter(2*torch.rand(self.weight_matrix_shape[1], 1, requires_grad=False)-1)
#         self.u.requires_grad = False
#         self.v.requires_grad = False
#
#     def _sync_device(self) -> None:
#         device = self.weight_matrix_fn().device
#         self.u = self.u.to(device)
#         self.v = self.v.to(device)
#
#     def step(self) -> None:
#         """
#         improve current approximation
#         """
#         self._sync_device()
#         with torch.no_grad():
#             W = self.weight_matrix_fn()
#             self.v.data = W.T @ self.u / torch.linalg.norm(W.T @ self.u)
#             self.u.data = W @ self.v / torch.linalg.norm(W @ self.v)
#
#     def get_approx(self) -> float:
#         self._sync_device()
#         return (self.u.T @ self.weight_matrix_fn() @ self.v).item()


class SpectralNormalizer(Normalizer):
    """
    Базовая спектральная нормализация
    """
    def __init__(self, module: T, beta: Optional[torch.Tensor] = None) -> None:
        """
        :param module:
        :param beta:
        """
        super().__init__(module)

        if isinstance(module, nn.Linear):
            self.weight_matrix_fn = lambda: module.weight.data
        elif isinstance(module, nn.Conv2d):
            self.weight_matrix_fn = lambda: module.weight.data.reshape(
                (module.weight.data.shape[0], -1))
        else:
            raise ModuleNotSupported

        self.weight_matrix_shape = self.weight_matrix_fn().shape
        u = 2*torch.rand(self.weight_matrix_shape[0], 1) - 1
        v = 2*torch.rand(self.weight_matrix_shape[1], 1) - 1
        self.register_buffer('u', u)
        self.register_buffer('v', v)

        self.beta = beta

    def _sync_device(self) -> None:
        device = self.weight_matrix_fn().device
        self.u = self.u.to(device)
        self.v = self.v.to(device)

    def calc_singular_value_approx(self) -> float:
        self._sync_device()
        return (self.u.T @ self.weight_matrix_fn() @ self.v).item()

    def update_stats(self, **kwargs) -> None:
        self._sync_device()
        with torch.no_grad():
            W = self.weight_matrix_fn()
            self.v.data = W.T @ self.u / torch.linalg.norm(W.T @ self.u)
            self.u.data = W @ self.v / torch.linalg.norm(W @ self.v)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        sing_approx = self.calc_singular_value_approx()
        output = self.module(X, *args, **kwargs)
        if self.beta is None:  # default strict mode
            return output / sing_approx
        elif sing_approx > self.beta:
            return self.beta * output / sing_approx
        else:
            return output


class WeakSpectralNormalizer(Normalizer):
    def __init__(self, module: T, beta: float, is_trainable_beta: bool = False):
        super().__init__(module)
        self.is_trainable_beta = is_trainable_beta
        if is_trainable_beta:
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)  # one for all layers
        else:
            self.register_buffer('beta', torch.tensor(beta))
        self.module = apply_normalization(self.module, SpectralNormalizer, beta=self.beta)


class MultiplyOutputNormalizer(Normalizer):
    def __init__(self, module: T, coef: float = 1., is_trainable_coef: bool = False):
        super().__init__(module)
        self.is_trainable_coef = is_trainable_coef
        if is_trainable_coef:
            self.coef = nn.Parameter(torch.tensor(coef), requires_grad=True)
        else:
            self.register_buffer('coef', torch.tensor(coef))

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.coef * self.module(X, *args, **kwargs)


def _get_final_submodules(module: nn.Module, res: List[nn.Module]) -> None:
    has_submodules = False
    for name, submodule in module.named_children():
        has_submodules = True
        _get_final_submodules(submodule, res)
    if not has_submodules:
        res.append(module)


def get_final_submodules(module: nn.Module) -> List[nn.Module]:
    res = []
    _get_final_submodules(module, res)
    return res


# used by 'MultiplyLayersOutputNormalizer'
class _MultiplyOutputNormalizer(Normalizer):
    def __init__(self, module: nn.Module, coefs: torch.Tensor, index: int):
        super().__init__(module)
        self.module = module
        self.coefs = coefs
        self.index = index

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.coefs[self.index] * self.module(X, *args, **kwargs)


def _modify_sublayers(module: nn.Module, func: Callable[[nn.Module], nn.Module], *args,
                      ids: Optional[FrozenSet[int]] = None, **kwargs) -> nn.Module:
    """
    if ids is None, then 'func' is applied only to leaf-modules
    """
    if ids is not None and id(module) in ids:
        return func(module, *args, **kwargs)
    is_leaf = True
    for name, submodule in module.named_children():
        is_leaf = False
        modified_submodule = _modify_sublayers(submodule, func, *args, ids=ids, **kwargs)
        setattr(module, name, modified_submodule)
    if is_leaf and ids is None:
        return func(module, *args, **kwargs)

    return module


def modify_sublayers(module: nn.Module, func: Callable[[nn.Module], nn.Module], *args,
                     layers: Optional[List[nn.Module]] = None, **kwargs) -> nn.Module:
    """
    replaces each layer listed in 'layers' by 'func'(layer)
    """
    ids = frozenset({id(layer) for layer in layers}) if layers is not None else None
    return _modify_sublayers(module, func, *args, ids=ids, **kwargs)


# as MultiplyOutputNormalizer but for multiple layers
class MultiplyLayersOutputNormalizer(Normalizer):
    def __init__(self, module: nn.Module, num_layers: Optional[int] = 1,
                 multiplied_layer_types: Optional[List[type]] = None,
                 layers: Optional[Iterable[nn.Module]] = None,
                 init_coef: Optional[float] = 1., init_prod_coef: Optional[float] = None,
                 is_trainable_coef: bool = False):
        """
        This normalizer supports two modes: "auto" and "manual":
        "manual" mode is active when 'layers' is not None:
            in this case the arguments 'num_layers' and 'multiplied_layer_types' are ignored
            and the output of each layer present in 'layers' is multiplied
        "auto" mode is active when 'layers' is None:
            in this case:
            'multiplied_layer_types' - types of layers to which multiplication may be applied
            'num_layers' - the number of layers to which multiplication is applied
                if 'num_layers' == 1, then the output of the entire net is multiplied
                if 'num_layers' == -1, then the output of all supported layers is multiplied
                NOTE: currently other values are not supported

            all nested modules are considered in this mode (the submodules of the 'module' that have no submodules)

        'init_coef' - a value by which the output of each chosen layer is multiplied
        'init_prod_coef' - mutually exclusive with 'init_coef', if specified, then it is the same as
            'init_coef' = 'init_prod_coef'^(1/n), where n is the number of chosen layers
        'is_trainable_coef' - if True, then all coefs are trainable parameters, otherwise they are not
        """
        super().__init__(module)

        def is_module_supported(module) -> bool:
            if multiplied_layer_types is None:
                return True
            else:
                is_supported = False
                for tp in multiplied_layer_types:
                    is_supported = is_supported or isinstance(module, tp)
                return is_supported

        if layers is None:  # auto mode
            assert num_layers is not None
            if num_layers == 1:
                layers = [module]
            elif num_layers == -1:
                all_layers = get_final_submodules(module)
                layers = [submodule for submodule in all_layers if is_module_supported(submodule)]
            else:
                raise NotImplementedError('Supported values for `num_values` are [-1, 1]')

        if init_coef is None:
            assert init_prod_coef is not None
            init_coef = math.pow(init_prod_coef, 1 / len(layers))
        else:
            assert init_prod_coef is None

        # 'init_coef' and 'layers' are defined here
        coefs = torch.full((len(layers),), init_coef)
        if is_trainable_coef:
            self.coefs = nn.Parameter(coefs, requires_grad=True)
        else:
            self.register_buffer('coefs', coefs)

        _appl_cnt = 0

        def apply_multiplier_to_layer(module: nn.Module):
            nonlocal _appl_cnt
            module = _MultiplyOutputNormalizer(module, coefs=self.coefs, index=_appl_cnt)
            _appl_cnt += 1
            return module

        module = modify_sublayers(module, func=apply_multiplier_to_layer, layers=layers)
        self.module = module

    def forward(self, X, *args, **kwargs):
        return self.module(X, *args, **kwargs)


def apply_normalization(module: nn.Module,
                        normalizer_cls: Type[Normalizer], *normalizer_args,
                        **normalizer_kwargs) -> nn.Module:
    try:
        return normalizer_cls(module, *normalizer_args, **normalizer_kwargs)
    except ModuleNotSupported:
        pass

    for name, submodule in module.named_children():
        normalized_submodule = apply_normalization(submodule, normalizer_cls, *normalizer_args,
                                                   **normalizer_kwargs)
        setattr(module, name, normalized_submodule)

    return module


def update_normalizers_stats(module: nn.Module, **kwargs):
    """
    Call when after a change of weights
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, Normalizer):
            submodule.update_stats(**kwargs)


from pipeline.device import get_local_device

# https://arxiv.org/pdf/2211.06595v1.pdf
class ABCASNormalizer(MultiplyLayersOutputNormalizer):
    def __init__(self, module: T, b: float = 4., alpha: float = 0.9999, m_const: float = 0.9,
                 r_start: float = 0.) -> None:
        module = apply_normalization(module, SpectralNormalizer)
        super().__init__(module=module, num_layers=-1, multiplied_layer_types=[nn.Linear, nn.Conv2d],
                         layers=None, init_coef=1., is_trainable_coef=False)
        self.b = b   # beta in the paper
        self.r = r_start  # this value is not used
        self.register_buffer('dm', torch.tensor(r_start))
        self.alpha = alpha
        self.m_const = m_const

    def update_stats(self, disc_real_vals: torch.Tensor, disc_gen_vals: torch.Tensor, **kwargs) -> None:
        super().update_stats()
        dist = (disc_real_vals.max() - disc_gen_vals.min()).item()
        self.dm = self.alpha * self.dm + (1 - self.alpha) * dist
        clbr_dm = self.dm / self.b
        self.r = max(0., clbr_dm / (1 - clbr_dm))
        m = self.m_const ** self.r
        self.coefs.fill_(m)
