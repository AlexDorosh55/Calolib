# calodiff.py
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import copy
import numpy as np
from typing import Callable, Optional, Dict, List, Tuple
from pipeline.metrics import *
from pipeline.custom_metrics import *
from pipeline.physical_metrics.calogan_prd import get_energy_embedding, calc_pr_rec_from_embeds, plot_pr_aucs
from pipeline.physical_metrics import calogan_metrics
from pipeline.physical_metrics.prd_score import compute_prd_from_embedding, prd_to_max_f_beta_pair


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # Compute query, key, value projections
        q = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # (B, H*W, C//8)
        k = self.key(x).view(batch_size, -1, H*W)  # (B, C//8, H*W)
        v = self.value(x).view(batch_size, -1, H*W)  # (B, C, H*W)
        
        # Compute attention map
        attention = torch.bmm(q, k)  # (B, H*W, H*W)
        attention = nn.functional.softmax(attention, dim=-1)
        
        # Apply attention to value and combine with residual
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x
        
class MixedConditionedUnet(nn.Module):
    def __init__(self, image_size=30, cond_emb_size=9):
        super().__init__()

        self.image_size = image_size  # Размер входного изображения (30x30)
        self.cond_emb_size = cond_emb_size  # Количество параметров условия

        # Преобразуем вектор условий [bs, 9] в тензор [bs, 128, 2, 2]
        self.fc1 = nn.Linear(cond_emb_size, 128 * 2 * 2)

        # Транспонированные сверточные слои, как в CaloganPhysicsGenerator
        self.conv1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.attn2 = SelfAttention(64)
        self.attn3 = SelfAttention(32)

        # UNet принимает изображение + доп. каналы
        self.model = UNet2DModel(
            sample_size=image_size + 2,  # Учитываем паддинг
            in_channels=2,  # 1 канал для изображения + 1 канал для условий
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t, condition):
        """
        x         : входное изображение [bs, 1, 30, 30]
        t         : временной шаг (для диффузии)
        condition : вектор условий [bs, 9]
        """
        bs, ch, w, h = x.shape
        # Добавляем паддинг, чтобы получить размер 32x32
        x = torch.nn.functional.pad(x, (1, 1, 1, 1))  # [bs, 1, 32, 32]

        # Кодируем вектор условий в тензор [bs, 128, 2, 2]
        condition_emb = nn.functional.relu(self.fc1(condition)).view(bs, 128, 2, 2)

        # Прогоняем через транспонированные сверточные слои
        EnergyDeposit = nn.functional.relu(self.bn1(self.conv1(condition_emb)))
        EnergyDeposit = nn.functional.relu(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposit = self.attn2(EnergyDeposit)
        EnergyDeposit = nn.functional.relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposit = self.attn3(EnergyDeposit)
        EnergyDeposit = nn.functional.relu(self.conv4(EnergyDeposit))

        # Обрезаем до [bs, 1, 30, 30]
        EnergyDeposit = EnergyDeposit[:, :, 1:31, 1:31]

        # Добавляем паддинг, чтобы получить размер 32x32
        EnergyDeposit = torch.nn.functional.pad(EnergyDeposit, (1, 1, 1, 1))  # [bs, 1, 32, 32]

        # Объединяем изображение и условие по каналу
        net_input = torch.cat((x, EnergyDeposit), 1)  # [bs, 2, 32, 32]

        # Прогоняем через UNet
        output = x - self.model(net_input, t).sample  # [bs, 1, 32, 32]

        # Обрезаем до [bs, 1, 30, 30] перед возвратом
        return output[:, :, 1:-1, 1:-1]

# --- Scheduler'ы шума для обучения ---

def _cosine_noise_scheduler(t: torch.Tensor, t_max: int) -> torch.Tensor:
    return 0.5 * (1 - torch.cos(torch.pi * t / t_max))

def _linear_noise_scheduler(t: torch.Tensor, t_max: int) -> torch.Tensor:
    return t / t_max

NOISE_SCHEDULERS = {
    "cosine": _cosine_noise_scheduler,
    "linear": _linear_noise_scheduler
}


# --- Функция инференса (генерации) ---

def sample(
        model: torch.nn.Module,
        y_conditions: torch.Tensor,
        n_steps: int,
        device: str,
        shape: tuple = (1, 30, 30),
        sampling_method: str = "default"
) -> torch.Tensor:
    """
    Функция для генерации изображений (инференса) с использованием обученной модели.
    """
    n_samples = y_conditions.shape[0]
    x_gen = torch.rand(n_samples, *shape).to(device)
    y_conditions = y_conditions.to(device)

    model.eval()
    with torch.no_grad():
        if sampling_method == "default":
            for i in tqdm(range(n_steps), desc="Sampling", leave=False):
                pred = model(x_gen, 0, y_conditions)
                mix_factor = 1 / (n_steps - i) if n_steps - i > 0 else 1.0
                x_gen = x_gen * (1 - mix_factor) + pred * mix_factor
        else:
            raise ValueError(f"Неизвестный метод сэмплинга: {sampling_method}")

    return x_gen.cpu()


# --- Основная функция обучения ---

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    valid_loader: Optional[DataLoader] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    noise_scheduler_name: str = "cosine",
    validation_freq: int = 1,
    n_inference_steps: int = 1000,
    checkpoint_path: str = "./checkpoints",
    early_stopping_patience: Optional[int] = None,
    test_loader: Optional[DataLoader] = None,
    visualize_test_batch: bool = True,
    test_visualization_func: Optional[Callable] = None
) -> Dict[str, List[float]]:
    """
    Универсальная функция для обучения диффузионной модели.
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    history = {'train_loss': [], 'valid_loss': []}
    best_train_loss = float('inf')
    best_model_state_on_train = None
    patience_counter = 0

    fixed_test_batch = None
    if test_loader and visualize_test_batch:
        try:
            fixed_test_batch = next(iter(test_loader))
        except StopIteration:
            print("Warning: test_loader пуст, визуализация на тестовом батче будет пропущена.")

    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1}/{n_epochs} ---")

        model.train()
        epoch_train_loss = []
        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, n_inference_steps, (x.shape[0],), device=device)
            noise_amount = noise_scheduler_fn(t.float(), n_inference_steps).view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            noisy_x = x * (1 - noise_amount) + noise * noise_amount
            pred = model(noisy_x, 0, y)
            loss = loss_fn(x, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())

        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        history['train_loss'].append(avg_train_loss)
        print(f"Avg Train Loss: {avg_train_loss:.5f}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_model_state_on_train = copy.deepcopy(model.state_dict())
            torch.save(best_model_state_on_train, os.path.join(checkpoint_path, "best_model_on_train.pth"))
            print(f"🚀 New best model saved with train loss: {best_train_loss:.5f}")

        if visualize_test_batch and fixed_test_batch is not None and test_visualization_func is not None:
            x_test_real, y_test = fixed_test_batch
            generated_images = sample(
                model, y_test, n_inference_steps, device,
                shape=x_test_real.shape[1:]
            )
            n_samples_to_show = min(len(generated_images), 5)
            fig, axs = plt.subplots(1, n_samples_to_show, figsize=(20, 4))
            fig.suptitle(f"Test Batch Visualization at Epoch {epoch + 1}", fontsize=16)
            if n_samples_to_show == 1: axs = [axs]
            for i, ax in enumerate(axs):
                test_visualization_func(energy=generated_images[i].cpu(), ax=ax)
            plt.show()

        if valid_loader and (epoch + 1) % validation_freq == 0:
            model.eval()
            epoch_valid_loss = []
            with torch.no_grad():
                for x_val, y_val in tqdm(valid_loader, desc="Validation"):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred_val = model(x_val, 0, y_val)
                    loss = loss_fn(x_val, pred_val)
                    epoch_valid_loss.append(loss.item())

            avg_valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
            history['valid_loss'].append(avg_valid_loss)
            print(f"Avg Validation Loss: {avg_valid_loss:.5f}")

            if lr_scheduler:
                lr_scheduler.step(avg_valid_loss) if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else lr_scheduler.step()

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
                print(f"✨ Validation loss improved to: {best_valid_loss:.5f}")
            elif early_stopping_patience:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Stopping early. No improvement in validation loss for {patience_counter} epochs.")
                    if best_model_state_on_train:
                        model.load_state_dict(best_model_state_on_train)
                    return history

    print("Training finished. Loading the best model based on training loss.")
    if best_model_state_on_train:
        model.load_state_dict(best_model_state_on_train)

    return history


# ===================================================================
# --- НОВЫЕ И ОБНОВЛЕННЫЕ ФУНКЦИИ ---
# ===================================================================

# --- 1. Функция инференса с сохранением ---

def inference_with_saving(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_steps: int,
    device: str,
    output_path: str = "generated_data.npz",
    sampling_method: str = "default"
):
    """
    Проводит инференс на всем даталоадере и сохраняет результаты в .npz файл.
    """
    all_real_images, all_gen_images, all_conditions = [], [], []
    model.to(device)
    model.eval()

    for x_real, y_cond in tqdm(dataloader, desc="Inference and Saving"):
        x_gen = sample(
            model, y_cond, n_steps, device,
            shape=x_real.shape[1:],
            sampling_method=sampling_method
        )
        all_real_images.append(x_real.cpu().numpy())
        all_gen_images.append(x_gen.cpu().numpy())
        all_conditions.append(y_cond.cpu().numpy())

    real_images_np = np.vstack(all_real_images)
    gen_images_np = np.vstack(all_gen_images)
    conditions_np = np.vstack(all_conditions)

    real_labels = np.ones(len(real_images_np))
    gen_labels = np.zeros(len(gen_images_np))

    final_images = np.vstack([real_images_np, gen_images_np])
    final_labels = np.hstack([real_labels, gen_labels])
    final_conditions = np.vstack([conditions_np, conditions_np])

    np.savez_compressed(
        output_path,
        images=final_images,
        labels=final_labels,
        conditions=final_conditions
    )
    print(f"✅ Данные успешно сохранены в файл: '{output_path}'")


# --- 2. Функция оценки и визуализации физических метрик ---

def _calculate_physics_metrics(
    gen_images: np.ndarray,
    real_images: np.ndarray,
    conditions: np.ndarray,
    num_clusters: int = 20
) -> Dict[str, np.ndarray]:
    """Вспомогательная функция для расчета физических метрик."""
    gen_images_sq = gen_images.reshape(-1, 30, 30) if gen_images.shape[1] == 1 else gen_images
    real_images_sq = real_images.reshape(-1, 30, 30) if real_images.shape[1] == 1 else real_images

    metrics = {
        "Gen Longitudual Asymmetry": calogan_metrics.get_assymetry(gen_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=False).flatten(),
        "Gen Transverse Asymmetry": calogan_metrics.get_assymetry(gen_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=True).flatten(),
        "Gen Longitudual Width": calogan_metrics.get_shower_width(gen_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=False).flatten(),
        "Gen Transverse Width": calogan_metrics.get_shower_width(gen_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=True).flatten(),
        "Real Longitudual Asymmetry": calogan_metrics.get_assymetry(real_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=False).flatten(),
        "Real Transverse Asymmetry": calogan_metrics.get_assymetry(real_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=True).flatten(),
        "Real Longitudual Width": calogan_metrics.get_shower_width(real_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=False).flatten(),
        "Real Transverse Width": calogan_metrics.get_shower_width(real_images_sq, conditions[:, 0:3], conditions[:, 6:], orthog=True).flatten(),
    }

    gen_physics_stats = np.stack([metrics["Gen Longitudual Asymmetry"], metrics["Gen Transverse Asymmetry"], metrics["Gen Longitudual Width"], metrics["Gen Transverse Width"]], axis=1)
    real_physics_stats = np.stack([metrics["Real Longitudual Asymmetry"], metrics["Real Transverse Asymmetry"], metrics["Real Longitudual Width"], metrics["Real Transverse Width"]], axis=1)

    precision_energy, recall_energy = calc_pr_rec_from_embeds(gen_images.reshape(gen_images.shape[0], -1), real_images.reshape(real_images.shape[0], -1), num_clusters=num_clusters)
    precision_physics, recall_physics = calc_pr_rec_from_embeds(gen_physics_stats, real_physics_stats, num_clusters=num_clusters)

    metrics.update({
        'PRD_energy_AUC': np.trapz(precision_energy, recall_energy),
        'precision_energy': precision_energy, 'recall_energy': recall_energy,
        'PRD_physics_AUC': np.trapz(precision_physics, recall_physics),
        'precision_physics': precision_physics, 'recall_physics': recall_physics
    })
    return metrics

def evaluate_and_visualize_physics_metrics(
    gen_images: torch.Tensor,
    real_images: torch.Tensor,
    conditions: torch.Tensor,
    num_clusters: int = 20,
    statistics_to_plot: List[str] = ['Longitudual Asymmetry', 'Transverse Asymmetry', 'Longitudual Width', 'Transverse Width']
):
    """
    Вычисляет и визуализирует физические метрики для сгенерированных и реальных изображений.
    """
    scores = _calculate_physics_metrics(
        gen_images.cpu().numpy(), real_images.cpu().numpy(), conditions.cpu().numpy(), num_clusters
    )

    print(f"--- Результаты Физических Метрик ---\nPRD Energy AUC: {np.mean(scores['PRD_energy_AUC']):.4f}\nPRD Physics AUC: {np.mean(scores['PRD_physics_AUC']):.4f}\n------------------------------------")
    sns.set_theme(style="whitegrid")

    for statistic in statistics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.histplot(scores['Gen ' + statistic], bins=50, alpha=0.6, label="Generated", color="orange", kde=True)
        sns.histplot(scores['Real ' + statistic], bins=50, alpha=0.6, label="Real", color="blue", kde=True)
        plt.title(f"Distribution of {statistic}", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.show()

    print('Energy PRD Curve')
    plot_pr_aucs(scores['precision_energy'], scores['recall_energy'])
    plt.show()

    print('Physics PRD Curve')
    plot_pr_aucs(scores['precision_physics'], scores['recall_physics'])
    plt.show()
    return scores


# --- 3. Функция оценки метрик на каждом шаге Denoising'а ---

def evaluate_metrics_over_denoising_steps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_steps: int,
    device: str,
    sampling_method: str = "default"
) -> Dict[str, List[float]]:
    """
    Оценивает изменение физических метрик на каждом шаге процесса denoising,
    используя один батч из даталоадера. Строит сглаженный график с областями
    стандартного отклонения.
    """
    model.to(device)
    model.eval()

    try:
        x_real, y_conditions = next(iter(dataloader))
    except StopIteration:
        print("Ошибка: dataloader пуст.")
        return {}

    x_real, y_conditions = x_real.to(device), y_conditions.to(device)
    x_gen = torch.rand_like(x_real)

    # Словари для хранения статистики на каждом шаге
    metrics_history = {
        'step': [],
        'energy_auc_mean': [], 'energy_auc_std': [],
        'physics_auc_mean': [], 'physics_auc_std': []
    }

    with torch.no_grad():
        if sampling_method == "default":
            for i in tqdm(range(n_steps), desc="Evaluating Denoising Steps"):
                pred = model(x_gen, 0, y_conditions)
                mix_factor = 1 / (n_steps - i) if n_steps - i > 0 else 1.0
                x_gen =  pred * 1.0

                current_metrics = _calculate_physics_metrics(
                    x_gen.cpu().numpy(), x_real.cpu().numpy(), y_conditions.cpu().numpy()
                )
                
                energy_auc = current_metrics['PRD_energy_AUC']
                physics_auc = current_metrics['PRD_physics_AUC']

                metrics_history['step'].append(i)
                metrics_history['energy_auc_mean'].append(np.mean(energy_auc))
                metrics_history['energy_auc_std'].append(np.std(energy_auc))
                metrics_history['physics_auc_mean'].append(np.mean(physics_auc))
                metrics_history['physics_auc_std'].append(np.std(physics_auc))
        else:
            raise ValueError(f"Неизвестный метод сэмплинга: {sampling_method}")

    print("✅ Анализ по шагам завершен.")

    # --- ✨ УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Извлекаем данные для удобства
    steps = np.array(metrics_history['step'])
    e_mean = np.array(metrics_history['energy_auc_mean'])
    e_std = np.array(metrics_history['energy_auc_std'])
    p_mean = np.array(metrics_history['physics_auc_mean'])
    p_std = np.array(metrics_history['physics_auc_std'])

    # Плот для Energy AUC
    ax.plot(steps, e_mean, label='PRD Energy AUC (Mean)', color='royalblue', linewidth=2)
    ax.fill_between(steps, e_mean - e_std, e_mean + e_std, color='royalblue', alpha=0.2, label='Std. Dev.')

    # Плот для Physics AUC
    ax.plot(steps, p_mean, label='PRD Physics AUC (Mean)', color='darkorange', linewidth=2)
    ax.fill_between(steps, p_mean - p_std, p_mean + p_std, color='darkorange', alpha=0.2)

    ax.set_xlabel("Denoising Step", fontsize=12)
    ax.set_ylabel("AUC Value", fontsize=12)
    ax.set_title("Изменение PRD AUC в процессе Denoising'а", fontsize=16)
    ax.legend()
    plt.show()

    return metrics_history
