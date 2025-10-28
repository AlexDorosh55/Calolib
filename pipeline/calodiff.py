# calodiff.py
# === Группа 1: Стандартные библиотеки Python ===
import os
import copy
from typing import Callable, Optional, Dict, List, Tuple

# === Группа 2: Сторонние библиотеки (Third-Party) ===

# PyTorch и связанные утилиты
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Научные вычисления и метрики
import numpy as np
from sklearn.metrics import auc

# Модели (Diffusers, THOP)
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel
from thop import profile as thop_profile

# Визуализация и прогресс-бар
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# === Группа 3: Локальные импорты (проект 'pipeline') ===
from pipeline.metrics import *
from pipeline.custom_metrics import *
from pipeline.physical_metrics import calogan_metrics
from pipeline.physical_metrics.calogan_prd import (
    get_energy_embedding, 
    calc_pr_rec_from_embeds, 
    plot_pr_aucs
)
from pipeline.physical_metrics.prd_score import (
    compute_prd_from_embedding, 
    prd_to_max_f_beta_pair
)


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
        noise_scheduler_fn: Callable,  
        shape: tuple = (1, 30, 30),
        sampling_method: str = "ddim"  
) -> torch.Tensor:
    """
    Функция для генерации изображений (инференса) - ИСПРАВЛЕННАЯ
    Добавлен метод 'ddim', который корректно работает с scheduler'ом
    """
    n_samples = y_conditions.shape[0]

    x_gen = torch.randn(n_samples, *shape).to(device)
    y_conditions = y_conditions.to(device)

    model.eval()
    with torch.no_grad():
        if sampling_method == "ddim":
            for i in tqdm(reversed(range(n_steps)), desc="Sampling (DDIM)", total=n_steps, leave=False):
                t_tensor = torch.full((n_samples,), i, device=device, dtype=torch.long)
                pred_x0 = model(x_gen, t_tensor, y_conditions)
                t_float = t_tensor.float()
                noise_amount_t = noise_scheduler_fn(t_float, n_steps).view(-1, 1, 1, 1)
                signal_amount_t = 1.0 - noise_amount_t
                t_prev_float = (t_float - 1).clamp(min=0)
                noise_amount_t_prev = noise_scheduler_fn(t_prev_float, n_steps).view(-1, 1, 1, 1)
                signal_amount_t_prev = 1.0 - noise_amount_t_prev
                pred_noise = (x_gen - signal_amount_t * pred_x0) / (noise_amount_t + 1e-8)
                x_gen = signal_amount_t_prev * pred_x0 + noise_amount_t_prev * pred_noise

        elif sampling_method == "default":
            print("Warning: Используется 'default' сэмплинг, который не связан с noise_scheduler'ом.")
            for i in tqdm(range(n_steps), desc="Sampling (Default)", leave=False):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)
                pred = model(x_gen, t, y_conditions)
                mix_factor = 1 / (n_steps - i) if n_steps - i > 0 else 1.0
                x_gen = x_gen * (1 - mix_factor) + pred * mix_factor
        else:
            raise ValueError(f"Неизвестный метод сэмплинга: {sampling_method}")

    return x_gen.cpu()
    
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
    Универсальная функция для обучения диффузионной модели. (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    history = {'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    best_model_state_on_valid = None 
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
            pred = model(noisy_x, t, y) 
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
            print(f"🚀 New best train model saved with train loss: {best_train_loss:.5f}")

        # Визуализация (здесь все было в порядке)
        if visualize_test_batch and fixed_test_batch is not None and test_visualization_func is not None:
            model.eval() 
            x_test_real, y_test = fixed_test_batch
            y_test = y_test.to(device) 
          
          # Генерируем изображения
            generated_images = sample(
                model, 
                y_test, 
                n_inference_steps, 
                device,
                noise_scheduler_fn, 
                shape=x_test_real.shape[1:],
                sampling_method="ddim" 
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
                    t_val = torch.randint(0, n_inference_steps, (x_val.shape[0],), device=device)
                    noise_amount_val = noise_scheduler_fn(t_val.float(), n_inference_steps).view(-1, 1, 1, 1)
                    noise_val = torch.randn_like(x_val)
                    noisy_x_val = x_val * (1 - noise_amount_val) + noise_val * noise_amount_val
                    pred_val = model(noisy_x_val, t_val, y_val)
                    loss = loss_fn(x_val, pred_val) 
                    
                    epoch_valid_loss.append(loss.item())

            avg_valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
            history['valid_loss'].append(avg_valid_loss)
            print(f"Avg Validation Loss: {avg_valid_loss:.5f}")

            if lr_scheduler:
                lr_scheduler.step(avg_valid_loss) if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else lr_scheduler.step()
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model_state_on_valid = copy.deepcopy(model.state_dict())
                torch.save(best_model_state_on_valid, os.path.join(checkpoint_path, "best_model_on_valid.pth"))
                print(f"New best model saved with validation loss: {best_valid_loss:.5f}")
                patience_counter = 0 
            elif early_stopping_patience:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"Stopping early. No improvement in validation loss for {patience_counter} epochs.")
                    if best_model_state_on_valid:
                        model.load_state_dict(best_model_state_on_valid)
                    return history

    print("Training finished.")
    if best_model_state_on_valid:
        print("Loading the best model based on validation loss.")
        model.load_state_dict(best_model_state_on_valid)
    elif best_model_state_on_train:
        print("Warning: No best validation model found. Loading best model on train loss.")
        model.load_state_dict(best_model_state_on_train)

    return history

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
    print(f"Данные успешно сохранены в файл: '{output_path}'")


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

def calculate_pr_metrics(precisions: List[np.ndarray], recalls: List[np.ndarray]):
    """
    Вычисляет PR-AUC для каждой кривой и стандартное отклонение точности
    по всем кривым. Не строит графики.

    Аргументы:
        precisions (List[np.ndarray]): Список массивов значений точности.
        recalls (List[np.ndarray]): Список массивов значений полноты.

    Возвращает:
        tuple: (pr_aucs, std_precisions)
            pr_aucs (List[float]): Список всех индивидуальных PR-AUC.
            std_precisions (np.ndarray): Стандартное отклонение точности (поэлементное).
    """

    pr_aucs = []
    for i in range(len(recalls)):
        pr_aucs.append(auc(precisions[i], recalls[i]))
    std_pr_aucs = np.std(pr_aucs, axis=0)

    return np.mean(pr_aucs), std_pr_aucs

def evaluate_metrics_over_denoising_steps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_steps: int,
    device: str,
    denoising_scheduler_name: str = "cosine"
) -> Dict[str, List[float]]:
    """
    Оценивает изменение физических метрик на каждом шаге процесса denoising.
    Обрабатывает данные по батчам, чтобы избежать переполнения памяти GPU.
    """
    model.to(device)
    model.eval()
    
    # Шаг 1: Загружаем все данные, но пока держим их на CPU
    all_x_real = []
    all_y_conditions = []
    for x_real_batch, y_conditions_batch in tqdm(dataloader, desc="Loading data to CPU"):
        all_x_real.append(x_real_batch)
        all_y_conditions.append(y_conditions_batch)

    if not all_x_real:
        print("Ошибка: dataloader пуст. Невозможно провести оценку.")
        return {}

    # Объединяем все батчи в единые тензоры на CPU
    x_real_cpu = torch.cat(all_x_real, dim=0)
    y_conditions_cpu = torch.cat(all_y_conditions, dim=0)
    
    n_samples = y_conditions_cpu.shape[0]
    shape = x_real_cpu.shape[1:]
    
    # Генерируем начальный шум тоже на CPU
    x_gen_cpu = torch.rand(n_samples, *shape)
    
    # Используем batch_size из оригинального dataloader'а
    batch_size = dataloader.batch_size

    metrics_history = {
        'step': [],
        'PRD_energy_AUC': [],
        'PRD_physics_AUC': [],
        'PRD_energy_AUC_std': [],
        'PRD_physics_AUC_std': []
    }

    with torch.no_grad():
        for i in tqdm(range(n_steps), desc="Evaluating Denoising Steps"):
            # Список для сбора результатов обработки батчей на текущем шаге
            generated_batches_for_step = []

            # Итерируемся по данным батчами, чтобы прогнать через модель
            for j in range(0, n_samples, batch_size):
                # Вырезаем текущий батч
                x_gen_batch = x_gen_cpu[j:j+batch_size].to(device)
                y_conditions_batch = y_conditions_cpu[j:j+batch_size].to(device)

                # Вызов модели только для одного батча!
                pred_batch = model(x_gen_batch, 0, y_conditions_batch)

                # Возвращаем результат на CPU и добавляем в список
                generated_batches_for_step.append(pred_batch.cpu())

                # Очистка памяти GPU (опционально, но помогает)
                del x_gen_batch, y_conditions_batch, pred_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Собираем результаты всех батчей в один тензор на CPU
            x_gen_cpu = torch.cat(generated_batches_for_step, dim=0)

            # Дальнейшие вычисления метрик, как и раньше
            gen_images_np = x_gen_cpu.numpy()
            real_images_np = x_real_cpu.numpy()
            conditions_np = y_conditions_cpu.numpy()

            current_metrics = _calculate_physics_metrics(gen_images_np, real_images_np, conditions_np)
            metrics_history['step'].append(i)
            current_prd_auc_energy, current_prd_auc_energy_std = calculate_pr_metrics(current_metrics['precision_energy'], current_metrics['recall_energy'])
            current_prd_auc_physics, current_prd_auc_physics_std = calculate_pr_metrics(current_metrics['precision_physics'], current_metrics['recall_physics'])
            
            metrics_history['PRD_energy_AUC'].append(current_prd_auc_energy)
            metrics_history['PRD_physics_AUC'].append(current_prd_auc_physics)
            metrics_history['PRD_energy_AUC_std'].append(current_prd_auc_energy_std)
            metrics_history['PRD_physics_AUC_std'].append(current_prd_auc_physics_std)

    print("Анализ по шагам завершен.")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_history['step'], metrics_history['PRD_energy_AUC'], label='PRD Energy AUC', marker='.')
    plt.fill_between(
        metrics_history['step'],
        [m - s for m, s in zip(metrics_history['PRD_energy_AUC'], metrics_history['PRD_energy_AUC_std'])],
        [m + s for m, s in zip(metrics_history['PRD_energy_AUC'], metrics_history['PRD_energy_AUC_std'])],
        alpha=0.2
    )
    plt.plot(metrics_history['step'], metrics_history['PRD_physics_AUC'], label='PRD Physics AUC', marker='.')
    plt.fill_between(
        metrics_history['step'],
        [m - s for m, s in zip(metrics_history['PRD_physics_AUC'], metrics_history['PRD_physics_AUC_std'])],
        [m + s for m, s in zip(metrics_history['PRD_physics_AUC'], metrics_history['PRD_physics_AUC_std'])],
        alpha=0.2
    )
    plt.xlabel("Denoising Step")
    plt.ylabel("AUC Value")
    plt.title("Изменение PRD AUC в процессе Denoising'а")
    plt.legend()
    plt.grid(True)
    plt.show()

    return metrics_history
def analyze_model_complexity(
    model: nn.Module,
    n_steps: int,
    batch_size: int = 8,
    image_size: int = 30,
    conditions_dim: int = 9,
    channels: int = 1,
    print_thop: bool = False,
    print_profiler: bool = False,
    print_data_gen: bool = False,
    verbose_thop: bool = False
) -> float:
    """
    Анализирует вычислительную сложность модели (GFLOPS, параметры, время)
    с использованием thop и PyTorch Profiler.

    Аргументы:
        model (nn.Module): Модель для анализа (например, 'net').
        n_steps (int): Количество шагов, используемое для расчета общих GFLOPS.
        batch_size (int, optional): Размер батча. По умолчанию 8.
        image_size (int, optional): Размер изображения (предполагается квадратное). По умолчанию 30.
        conditions_dim (int, optional): Размерность вектора условий (y). По умолчанию 9.
        channels (int, optional): Количество входных каналов изображения. По умолчанию 1.
        print_thop (bool, optional): Печатать ли сводку thop (GFLOPS, параметры). По умолчанию False.
        print_profiler (bool, optional): Печатать ли детальный отчет PyTorch Profiler. По умолчанию False.
        print_data_gen (bool, optional): Печатать ли анализ затрат на генерацию данных. По умолчанию False.
        verbose_thop (bool, optional): Включить ли подробный вывод от самой thop. По умолчанию False.

    Возвращает:
        float: Суммарные GFLOPS на один батч (GFLOPS одного прохода * n_steps).
               Возвращает -1.0 в случае ошибки thop.
    """

    # --- 0. Настройка устройства и модели ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Важно для анализа, отключает dropout и т.д.

    # Создание фиктивных данных
    try:
        dummy_x = torch.randn(batch_size, channels, image_size, image_size).to(device)
        dummy_y = torch.randn(batch_size, conditions_dim).float().to(device)
        dummy_t = 0  # Пример временного шага (можно использовать torch.randint)
        
        # Входы для thop в виде кортежа
        thop_inputs = (dummy_x, dummy_t, dummy_y)
    except Exception as e:
        print(f"Ошибка при создании фиктивных тензоров: {e}")
        return -1.0

    # --- 1. Общий анализ с помощью thop ---
    total_gflops_per_batch = -1.0  # Значение по умолчанию в случае ошибки
    try:
        # thop_profile может не поддерживать некоторые операции
        macs, params = thop_profile(model, inputs=thop_inputs, verbose=verbose_thop)
        gflops = 2 * macs / 1e9
        total_gflops_per_batch = gflops * n_steps

        if print_thop:
            print("\n" + "="*50)
            print("### 1. ОБЩИЙ АНАЛИЗ (THOP) ###")
            print("="*50)
            print(f"Модель '{type(model).__name__}' выполняет {gflops:.2f} GFLOPS за один проход.")
            print(f"Количество параметров: {params / 1e6:.2f} M")
            print(f"Суммарные вычисления на один батч ({n_steps} шагов): {total_gflops_per_batch:.2f} GFLOPS")

    except Exception as e:
        if print_thop:
            print("\n" + "="*50)
            print("### 1. ОБЩИЙ АНАЛИЗ (THOP) - ОШИБКА ###")
            print("="*50)
            print(f"Не удалось выполнить анализ thop: {e}")
            print("Проверьте, поддерживает ли thop все операции в вашей модели.")


    # --- 2. Детальный анализ по слоям с помощью PyTorch Profiler ---
    if print_profiler:
        print("\n" + "="*50)
        print("### 2. ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛИ ###")
        print("="*50)

        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        try:
            with profile(
                activities=activities,
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    model(*thop_inputs)
            
            sort_key = "cuda_time_total" if device.type == 'cuda' else "cpu_time_total"
            
            print(f"--- Топ 15 операций по времени выполнения ({device.type}) ---")
            print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

            if device.type == 'cuda':
                print("\n--- Топ 15 операций по использованию памяти на GPU ---")
                print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))

        except Exception as e:
            print(f"Ошибка при выполнении PyTorch Profiler: {e}")


    # --- 3. Анализ затрат на генерацию данных ---
    if print_data_gen:
        print("\n" + "="*50)
        print("### 3. АНАЛИЗ ЗАТРАТ НА ГЕНЕРАЦИЮ ДАННЫХ ###")
        print("="*50)

        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
            
        try:
            with profile(activities=activities) as prof_data:
                _ = torch.randn(batch_size, channels, image_size, image_size).to(device)
                _ = torch.randn(batch_size, conditions_dim).float().to(device)

            sort_key = "cuda_time_total" if device.type == 'cuda' else "cpu_time_total"
            print(f"--- Затраты на создание и перемещение тензоров ({device.type}) ---")
            print(prof_data.key_averages().table(sort_by=sort_key, row_limit=10))
        except Exception as e:
             print(f"Ошибка при профилировании генерации данных: {e}")

    # --- 4. Возврат основного значения ---
    return total_gflops_per_batch
