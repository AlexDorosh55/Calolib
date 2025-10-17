from abc import abstractmethod
from typing import Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn

from pipeline import aux


class Discriminator(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param x: object from the considered space
        :param y: condition
        None means no condition.
        A discriminator knows the exact type of condition and how to use it.
        If discriminator does not support conditions, it is expected to raise an exception.
        """
        pass


def save_dimensions_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    works only for odd kernel size values
    returns padding size such that the output has the same coordinate dimensions
    """
    res = []
    for sz in kernel_size:
        if sz % 2 == 0:
            raise ValueError('Only odd kernel size values are supported')
        res.append((sz - 1) // 2)
    return tuple(res)


# взят из репозитория https://github.com/LucaAmbrogioni/Wasserstein-GAN-on-MNIST/blob/master/Wasserstein%20GAN%20playground.ipynb
class MNISTDiscriminator(Discriminator):
    def __init__(self, condition_classes_cnt: int = 0):
        super().__init__()
        self.condition_classes_cnt = condition_classes_cnt

        y_out = 0
        if condition_classes_cnt != 0:
            y_out = 10  # размерность вектора, в который переводится y (ohe)
            self.y_transform = nn.Identity()
            # self.y_transform = nn.Linear(in_features=condition_classes_cnt, out_features=y_out)

        self.x_to_vector = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(in_features=3*3*512 + y_out, out_features=1)

    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        x_vec = self.x_to_vector(x)
        if y is not None:
            assert isinstance(y, torch.Tensor)
            y = aux.ohe_labels(y, self.condition_classes_cnt)
            y_vec = self.y_transform(y)
            x_vec = torch.concat((x_vec, y_vec), dim=1)

        return self.fc(x_vec)


class MlpMnistDiscriminator(Discriminator):
    def __init__(self, condition_classes_cnt: int = 0):
        super().__init__()
        self.condition_classes_cnt = condition_classes_cnt

        mlp_in_len = 28*28
        if self.condition_classes_cnt != 0:
            class_embedding_len = 10
            self.embeddings = nn.Embedding(condition_classes_cnt, class_embedding_len)
            mlp_in_len += class_embedding_len

        self.mlp = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        x = x.reshape(x.shape[0], 28*28)
        if y is not None:
            y_embed = self.embeddings(y.long())
            x = torch.concat([x, y_embed], dim=1)
        return self.mlp(x)


class SimpleImageDiscriminator(Discriminator):  # for (1 x 28 x 28) images
    def __init__(self):
        super().__init__()

        # backbone
        conv_channels = 28

        self.backbone_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3, 3),
                      padding=save_dimensions_padding((3, 3))),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3),
                      padding=save_dimensions_padding((3, 3))),
            nn.BatchNorm2d(num_features=conv_channels),
        )

        self.backbone_end = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),  # -> conv_channels x 4 x 4
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=conv_channels * 4 * 4, out_features=1)
        )

    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        if y is not None:
            raise RuntimeError('Discriminator does not support condition')
        backbone_seq_out = self.backbone_seq(x)
        backbone_out = self.backbone_end(backbone_seq_out)
        out = self.head(backbone_out)
        return out


class SimplePhysicsDiscriminator(Discriminator):
    def __init__(self):
        super().__init__()
        
#         energy_dim = 30
        point_dim = 2
        momentum_dim = 3
        in_matr_dim = 10
        
        self.x_transform = nn.Sequential(  # 1x28x28 -> 5x10x10
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=4, stride=1),
            nn.BatchNorm2d(num_features=5),
            nn.ReLU(),
        )
        
        self.point_to_matr = nn.Sequential(
            nn.Linear(in_features=point_dim, out_features=in_matr_dim**2),
            nn.Unflatten(1, unflattened_size=(1, in_matr_dim, in_matr_dim))
        )
        
        self.momentum_to_matr = nn.Sequential(
            nn.Linear(in_features=momentum_dim, out_features=in_matr_dim**2),
            nn.Unflatten(1, unflattened_size=(1, in_matr_dim, in_matr_dim))
        )
        
        self.tensor_transform = nn.Sequential(  # 7x10x10
            nn.Conv2d(in_channels=7, out_channels=10, kernel_size=4),  # 10x7x7
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=4, dilation=2),  # 15x1x1
            nn.ReLU(),
            nn.Flatten(),
        )
        # 15
        self.head = nn.Linear(in_features=15, out_features=1)
        
    def forward(self, x: torch.Tensor, y):
        point, momentum = y
        x_matr = self.x_transform(x)
        point_matr = self.point_to_matr(point)
        momentum_matr = self.momentum_to_matr(momentum)
        stacked_matrs = torch.concat([x_matr, point_matr, momentum_matr], dim=1)
        in_tensor = nn.ReLU()(stacked_matrs)
        
        res = self.tensor_transform(in_tensor)
        res = res.reshape((res.shape[0], -1))
        c = self.head(res)
        return c


class CaloganPhysicsDiscriminator(Discriminator):
    def __init__(self, act_func=F.leaky_relu, add_points_norms_and_angles: bool = True):
        super().__init__()
        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        # 30x30x1 -> 32x32x1 (padding)
        # 32x32x1
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        # 16x16x32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        # 8x8x64
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=0)
        # 4x4x128
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=0)
        # 2x2x256
        # self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=0)
        # 1x1x256

        condition_dim = 7 if add_points_norms_and_angles else 5
        self.fc1 = nn.Linear(256 + condition_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, EnergyDeposit, y):
        point, momentum = y
        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        # print(EnergyDeposit.shape)
        X = self.activation(self.conv1(EnergyDeposit))
        # print(X.shape)
        X = self.activation(self.conv2(X))
        # print(X.shape)
        X = self.activation(self.conv3(X))
        # print(X.shape)
        X = self.activation(self.conv4(X))
        # print(X.shape)

        X = X.reshape(-1, 256)
        # print(X.shape)
        X = torch.cat([X, momentum, point], dim=1)
        # print(X.shape)

        X = F.leaky_relu(self.fc1(X))
        # print(X.shape)
        X = F.leaky_relu(self.fc2(X))
        # print(X.shape)

        return self.fc3(X)


class SmallCaloganPhysicsDiscriminator(Discriminator):
    def __init__(self, act_func=F.leaky_relu, add_points_norms_and_angles: bool = True):
        super().__init__()
        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        # 30x30x1 -> 32x32x1 (padding)
        # 32x32x1
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        # 16x16x32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        # 8x8x64
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        # 4x4x128
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=0)
        # 2x2x256
        # self.conv5 = nn.Conv2d(128, 128, 3, stride=2, padding=0)
        # 1x1x256

        condition_dim = 7 if add_points_norms_and_angles else 5
        self.fc1 = nn.Linear(128 + condition_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, EnergyDeposit, y):
        point, momentum = y
        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        # print(EnergyDeposit.shape)
        X = self.activation(self.conv1(EnergyDeposit))
        # print(X.shape)
        X = self.activation(self.conv2(X))
        # print(X.shape)
        X = self.activation(self.conv3(X))
        # print(X.shape)
        X = self.activation(self.conv4(X))
        # print(X.shape)

        X = X.reshape(-1, 128)
        # print(X.shape)
        X = torch.cat([X, momentum, point], dim=1)
        # print(X.shape)

        X = F.leaky_relu(self.fc1(X))
        # print(X.shape)
        X = F.leaky_relu(self.fc2(X))
        # print(X.shape)

        return self.fc3(X)
