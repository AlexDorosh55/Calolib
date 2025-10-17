# Taken from: https://github.com/SchattenGenie/mlhep2019_2_phase/blob/master/analysis
import pathlib
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from torch import nn
from tqdm import tqdm

from .prd_score import compute_prd_from_embedding


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.batchnorm0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, 2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 2, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2)

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(256, 256)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2 + 3)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.batchnorm0(self.dropout(x))
        x = self.batchnorm1(self.dropout(F.relu(self.conv1(x))))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))  # 64, 5, 5
        x = x.view(len(x), -1)
        x = self.dropout(x)
        x = self.batchnorm4(self.dropout(F.relu(self.fc1(x))))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x), self.fc5(x)

    def get_encoding(self, x):
        x = self.batchnorm0(self.dropout(x))
        x = self.batchnorm1(self.dropout(F.relu(self.conv1(x))))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))  # 64, 5, 5
        x = x.view(len(x), -1)
        x = self.dropout(x)
        x = self.batchnorm4(self.dropout(F.relu(self.fc1(x))))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_embedder(state_path: str):
    embedder = Regressor()
    state_dict = torch.load(state_path)
    embedder.load_state_dict(state_dict)
    embedder.eval()
    return embedder


embedder_state_path = pathlib.Path(__file__).parent / pathlib.Path('./embedder_state.pt')
embedder = load_embedder(str(embedder_state_path))


def get_energy_embedding(data):
    return embedder.get_encoding(data[0].view(-1, 1, 30, 30)).detach().numpy(), data[1]


def check_tensor_is_finite(t: np.ndarray) -> torch.Tensor:
    x = t.astype(np.float64)
    # for sklearn
    if np.isnan(x).sum() > 0:
        print('tensor contains NaN')
    if np.isinf(x).sum() > 0:
        print('tensor contains inf')
    if np.isneginf(x).sum() > 0:
        print('tensor contains -inf')
    mask = np.isnan(x) | np.isinf(x) | np.isneginf(x)

    bad_rows = mask.sum(axis=1) != 0
    bad_rows_cnt = bad_rows.sum()
    if bad_rows_cnt != 0:
        print(f'{bad_rows_cnt} bad rows in tensor')
    return x[~bad_rows]


def calc_pr_rec_from_embeds(data_real_embeds: np.ndarray, data_fake_embeds: np.ndarray, num_clusters=20, num_runs=10, NUM_RUNS=10,
                            show_progress_bar: bool = False,
                            enforce_balance: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data_real_embeds = check_tensor_is_finite(data_real_embeds)
    data_fake_embeds = check_tensor_is_finite(data_fake_embeds)

    precisions = []
    recalls = []
    for _ in tqdm(range(NUM_RUNS)) if show_progress_bar else range(NUM_RUNS):
        precision, recall = compute_prd_from_embedding(data_real_embeds, data_fake_embeds,
                                                       num_clusters=num_clusters, num_runs=num_runs,
                                                       enforce_balance=enforce_balance)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def plot_pr_aucs(precisions: List[np.ndarray], recalls: List[np.ndarray]):
    plt.figure(figsize=(12, 12))
    pr_aucs = []  # list of all pr-auc values
    for i in range(len(recalls)):
        plt.step(recalls[i], precisions[i], color='b', alpha=0.2)
        pr_aucs.append(auc(precisions[i], recalls[i]))
    plt.step(np.mean(recalls, axis=0), np.mean(precisions, axis=0), color='r', alpha=1,
             label=f'average, PR-AUC={np.mean(pr_aucs):.4f}')
    plt.fill_between(np.mean(recalls, axis=0),
                     np.mean(precisions, axis=0) - np.std(precisions, axis=0) * 3,
                     np.mean(precisions, axis=0) + np.std(precisions, axis=0) * 3, color='g',
                     alpha=0.2, label='std')

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # print(np.mean(pr_aucs), np.std(pr_aucs))
    plt.legend()
    plt.title('PRD')

    return pr_aucs
