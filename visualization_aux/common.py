import numpy as np
import matplotlib.pyplot as plt
import torch

from pipeline.device import get_local_device
from pipeline.gan import GAN


def imshow(img, ax=None, cmap='viridis', affine: bool = True):
    npimg = img.detach().numpy()
    if affine:  # если изначально к изображениям применялось это преобразование
        npimg = npimg / 2 + 0.5  # обратное афинное преобразование
    if ax is None:
        if npimg.shape[0] == 1:  # 1 channel
            plt.imshow(np.squeeze(npimg), cmap=cmap)
        else:  # 3 channels
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
        plt.show()
    else:
        if npimg.shape[0] == 1:  # 1 channel
            ax.imshow(np.squeeze(npimg), cmap=cmap)
        else:  # 3 channels
            ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)


def gen_several_images(gan_model: GAN, n: int = 5, y=None, figsize=(13, 13), imshow_fn=imshow):
    """
    Выводит n изображений, сгенерированных gan_model в строке
    """
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
    gan_model.to(get_local_device())
    with torch.no_grad():
        noise_batch = gan_model.gen_noise(n).to(get_local_device())
        gen_batch = gan_model.generator(noise_batch, y)

    if n == 1:
        axs = [axs]
    for i, (tensor, ax) in enumerate(zip(gen_batch, axs)):
        imshow_fn(tensor.cpu(), ax=ax)
        if y is not None and isinstance(y, torch.Tensor):
            ax.set_xlabel(y[i].item())

    plt.show()
