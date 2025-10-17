from typing import Callable, Dict, Any

import torch
from torch import nn

from pipeline.generators import Generator
from pipeline.discriminators import Discriminator


class GAN(nn.Module):
    def __init__(self, generator: Generator, discriminator: Discriminator,
                 noise_generator: Callable[[int], torch.Tensor]) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator

    def gen_noise(self, n: int) -> torch.Tensor:
        return self.noise_generator(n)

    def forward(self, noise=None):
        noise = noise or self.gen_noise(1)
        return self.generator(noise)

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        return {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self.generator.load_state_dict(state_dict['generator'])
        self.discriminator.load_state_dict(state_dict['discriminator'])

    def to(self, device) -> 'GAN':
        self.generator.to(device)
        self.discriminator.to(device)
        return self
