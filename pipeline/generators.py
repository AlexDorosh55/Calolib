from abc import abstractmethod
from typing import Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pipeline import aux


class Generator(nn.Module):
    @abstractmethod
    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param z: seed/noise for generation
        :param y: condition
        None means no condition.
        A generator knows the exact type of condition and how to use it for generation.
        If generator does not support conditions, it is expected to raise an exception.
        """
        pass


# взят из репозитория https://github.com/LucaAmbrogioni/Wasserstein-GAN-on-MNIST/blob/master/Wasserstein%20GAN%20playground.ipynb
class MNISTGenerator(Generator):
    def __init__(self, noise_dim: int, condition_classes_cnt: int = 0):
        """
        Uses one-hot-encoded label as optional condition
        0 means no condition
        """
        self.condition_classes_cnt = condition_classes_cnt

        super().__init__()

        conv_channels = 512
        base_width = 3

        if condition_classes_cnt != 0:
            y_out = 10  # размерность вектора, в который переводится y (ohe)
            noise_out = 50

            # self.noise_transform = nn.Linear(in_features=noise_dim, out_features=noise_out)
            self.noise_transform = nn.Identity()
            # self.y_transform = nn.Linear(in_features=condition_classes_cnt, out_features=y_out)
            self.y_transform = nn.Identity()
        else:
            self.noise_transform = nn.Identity()
            y_out = 0
            noise_out = noise_dim

        # 1 x (noise_out + y_out) -> 28 x 28
        self.model = nn.Sequential(
            nn.Linear(in_features=noise_out + y_out,
                      out_features=base_width * base_width * conv_channels),
            nn.Unflatten(dim=1, unflattened_size=(conv_channels, base_width, base_width)),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels, out_channels=conv_channels // 2,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 2, out_channels=conv_channels // 4,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 4, out_channels=conv_channels // 8,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 8, out_channels=1, kernel_size=3,
                               stride=3, padding=1),
            nn.Tanh(),
            # nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param y: integer labels
        """
        z = self.noise_transform(z)
        if y is not None:
            assert isinstance(y, torch.Tensor)
            # apply one-hot-encoding
            y_vec = aux.ohe_labels(y, self.condition_classes_cnt)
            y_trans = self.y_transform(y_vec)
            z = torch.concat((z, y_trans), dim=1)
        x = self.model(z)
        return x


# taken from https://github.com/arturml/mnist-cgan/blob/master/mnist-cgan.ipynb
class MlpMnistGenerator(Generator):
    def __init__(self, noise_dim: int, condition_classes_cnt: int = 0):
        super().__init__()
        self.condition_classes_cnt = condition_classes_cnt

        mlp_in_len = noise_dim
        if condition_classes_cnt != 0:
            class_embedding_len = 10
            self.embeddings = nn.Embedding(condition_classes_cnt, class_embedding_len)
            mlp_in_len += class_embedding_len

        self.mlp = nn.Sequential(
            nn.Linear(in_features=mlp_in_len, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=28*28),
        )

    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        if y is not None:
            y_embed = self.embeddings(y.long())
            z = torch.concat([z, y_embed], dim=1)
        res = self.mlp(z)
        batch_size = res.shape[0]
        return res.reshape(batch_size, 1, 28, 28)


class SimpleImageGenerator(Generator):
    def __init__(self, noise_dim: int, output_shape: Tuple[int, ...]):
        super().__init__()
        output_len = int(np.prod(output_shape))
        hidden_neurons = (noise_dim + output_len) // 2
        self.model = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=hidden_neurons),
            nn.ReLU(),
            nn.Linear(in_features=hidden_neurons, out_features=output_len),
        )
        self.output_shape = output_shape

    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        if y is not None:
            raise RuntimeError('Generator does not support condition')
        x = self.model(z)
        batch_size = z.shape[0]
        return x.reshape(batch_size, *self.output_shape)


class SimplePhysicsGenerator(Generator):
    def __init__(self, noise_dim: int):
        super().__init__()
        
        point_dim = 2
        momentum_dim = 3
        
        in_matr_dim = 10
        
        self.noise_to_matr = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=in_matr_dim**2),
            nn.Unflatten(1, unflattened_size=(1, in_matr_dim, in_matr_dim))
        )
        
        self.point_to_matr = nn.Sequential(
            nn.Linear(in_features=point_dim, out_features=in_matr_dim**2),
            nn.Unflatten(1, unflattened_size=(1, in_matr_dim, in_matr_dim))
        )
        
        self.momentum_to_matr = nn.Sequential(
            nn.Linear(in_features=momentum_dim, out_features=in_matr_dim**2),
            nn.Unflatten(1, unflattened_size=(1, in_matr_dim, in_matr_dim))
        )
        
        # 3 x 10 x 10
        self.tensor_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=5, kernel_size=4, stride=2, padding=1),  # 5 x 20 x 20
            nn.BatchNorm2d(num_features=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=10, kernel_size=4, stride=2, padding=6),  # 10 x 30 x 30
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1),  # 1 x 30 x 30
            nn.ReLU(),
        )
        
    def forward(self, z: torch.Tensor, y) -> torch.Tensor:
        point, momentum = y
        
        noise_matr = self.noise_to_matr(z)
        point_matr = self.point_to_matr(point)
        momentum_matr = self.momentum_to_matr(momentum)
        stacked_matrs = torch.concat([noise_matr, point_matr, momentum_matr], dim=1)
        in_tensor = nn.ReLU()(stacked_matrs)
        res = self.tensor_transform(in_tensor)
        return res


class CaloganPhysicsGenerator(Generator):
    def __init__(self, noise_dim: int, act_func=F.relu, add_points_norms_and_angles: bool = True):
        super().__init__()
        self.noise_dim = noise_dim
        self.activation = act_func

        self.add_points_norms_and_angles = add_points_norms_and_angles

        # 128 + 5 -> (+reshape) 128 x 2 x 2
        condition_dim = 7 if add_points_norms_and_angles else 5
        self.fc1 = nn.Linear(self.noise_dim + condition_dim, self.noise_dim * 2 * 2)

        # Z x 2 x 2
        self.conv1 = nn.ConvTranspose2d(self.noise_dim, 128, 3, stride=2, padding=1, output_padding=1)
        # 128 x 8 x 8
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        # 64 x 16 x 16
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        # 32 x 32 x 32
        self.conv4 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        # 1 x 32 x 32
        # crop
        # 1 x 30 x 30

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, z: torch.Tensor, y) -> torch.Tensor:
        point, momentum = y
        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        x = torch.cat([z, momentum, point], dim=1)
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x))

        # print(x.shape)
        EnergyDeposit = x.view(-1, self.noise_dim, 2, 2)

        # print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn1(self.conv1(EnergyDeposit)))
        # print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn2(self.conv2(EnergyDeposit)))
        # print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn3(self.conv3(EnergyDeposit)))
        # print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.conv4(EnergyDeposit))
        # print(EnergyDeposit.shape)
        EnergyDeposit = EnergyDeposit[:, :, 1:31, 1:31]
        # print(EnergyDeposit.shape)

        return EnergyDeposit
