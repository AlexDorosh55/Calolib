import torch


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def get_local_device() -> torch.device:
    return device
