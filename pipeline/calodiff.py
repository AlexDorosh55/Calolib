# calodiff.py
# === Группа 1: Стандартные библиотеки Python ===
import os
import copy
from typing import Callable, Optional, Dict, List, Tuple, Union

# === Группа 2: Сторонние библиотеки (Third-Party) ===

# PyTorch и связанные утилиты
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

# Научные вычисления и метрики
import numpy as np
import pandas as pd
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
def get_coefficients(alpha_t, sigma_t, alpha_prev, sigma_prev):
    """
    Вычисляет lambda (log-SNR) и шаг h.
    Добавлена защита от log(0).
    """
    # Защита от очень маленьких sigma (log(0) -> -inf)
    # Используем 1e-12 как минимум, чтобы избежать NaN в log
    sigma_t_safe = torch.clamp(sigma_t, min=1e-12)
    sigma_prev_safe = torch.clamp(sigma_prev, min=1e-12)
    
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t_safe)
    lambda_prev = torch.log(alpha_prev) - torch.log(sigma_prev_safe)
    
    h = lambda_prev - lambda_t
    return lambda_t, lambda_prev, h

def multistep_dpm_solver_update(x, model_out, history, alpha_t, sigma_t, alpha_prev, sigma_prev, order=2):
    """
    DPM-Solver++ (2M) с защитой от деления на ноль.
    """
    lambda_t, lambda_prev, h = get_coefficients(alpha_t, sigma_t, alpha_prev, sigma_prev)
    
    # phi_1(h) = exp(-h) - 1
    # Если h -> inf (последний шаг), то exp(-h) -> 0.
    phi_1 = torch.expm1(-h) # Более точное вычисление (exp(x) - 1)
    
    # DPM-Solver++ 2M Step 1 (First Order part)
    # x_{t-1} = (sigma_{t-1} / sigma_t) * x_t - alpha_{t-1} * phi_1 * x0_hat
    
    # Если sigma_t слишком мал (начало генерации из чистого), избегаем деления
    if torch.any(sigma_t < 1e-8):
         scale_term = torch.ones_like(x) # fallback (маловероятно в середине процесса)
    else:
         scale_term = sigma_prev / sigma_t

    D1 = scale_term * x - alpha_prev * phi_1 * model_out

    # Если это первый шаг или мы принудительно просим 1-й порядок (например, последний шаг), возвращаем D1
    if order == 1 or len(history) < 1:
        return D1
    
    # --- Second Order Part ---
    # DPM-Solver++ 2M
    
    m_last = history[-1] # (model_out_prev, lambda_prev)
    h_last = lambda_t - m_last[1] # Шаг между текущим и прошлым
    
    # Расчет r = h_last / h
    # Если h очень большой (последний шаг) или 0, будет ошибка.
    # Заменяем деление на безопасное
    
    # Если h очень велико (inf), r -> 0.
    # Если h близко к 0, r может быть большим.
    
    # Маска для безопасности (если h < 1e-5 или inf, отключаем 2-й порядок)
    valid_h_mask = (torch.abs(h) > 1e-5) & (torch.abs(h) < 1e5)
    
    # Если h некорректен, просто возвращаем D1 (fallback to 1st order)
    if not torch.all(valid_h_mask):
        return D1

    r = h_last / h
    
    # D2 term: alpha_prev * phi_1 / (2*r) * (model_out - m_last[0])
    # Если r -> 0, это взрывается. Защита:
    r = torch.clamp(r, min=1e-4) # Избегаем деления на 0
    
    denom = 2 * r
    c1 = 1 + 1 / denom
    c2 = -1 / denom
    
    x0_combined = c1 * model_out + c2 * m_last[0]
    
    x_prev = scale_term * x - alpha_prev * phi_1 * x0_combined
    return x_prev

def sample(
        model: torch.nn.Module,
        y_conditions: torch.Tensor,
        n_steps: int,
        device: str,
        noise_scheduler_fn: Callable,
        shape: tuple = (1, 30, 30),
        sampling_method: str = "dpm++", 
        cache_interval: int = 1,
        compute_steps_schedule: Optional[List[int]] = None, # <--- 1. Новый аргумент
        return_all_steps: bool = False,
        specific_steps: Optional[List[int]] = None
) -> torch.Tensor:

    n_samples = y_conditions.shape[0]
    x_gen = torch.randn(n_samples, *shape).to(device)
    y_conditions = y_conditions.to(device)

    model.eval()
    history_buffer = [] 
    
    # Инициализация кэша
    cached_model_out = None # <--- 2. Переменная для кэша

    if specific_steps is not None:
        timesteps = sorted(specific_steps, reverse=True)
    else:
        timesteps = list(reversed(range(n_steps)))

    with torch.no_grad():
        for i, t_curr in enumerate(timesteps):
            # Проверяем, является ли этот шаг последним
            is_last_step = (i == len(timesteps) - 1)
            
            if not is_last_step:
                t_prev = timesteps[i+1]
            else:
                t_prev = -1 
            
            # --- 3. Логика решения: Вычислять или Кэшировать? ---
            should_compute = False

            # Всегда вычисляем на самом первом шаге, чтобы заполнить кэш
            if i == 0:
                should_compute = True
            elif compute_steps_schedule is not None:
                # Если задан конкретный список шагов, проверяем наличие текущего t в нем
                if t_curr in compute_steps_schedule:
                    should_compute = True
            else:
                # Иначе используем интервал (старое поведение)
                if i % cache_interval == 0:
                    should_compute = True

            # --- Выполнение ---
            if should_compute:
                t_tensor = torch.full((n_samples,), t_curr, device=device, dtype=torch.long)
                model_out = model(x_gen, t_tensor, y_conditions) 
                cached_model_out = model_out # Обновляем кэш
            else:
                # Используем результат с предыдущего вычисления
                model_out = cached_model_out

            # --- Далее стандартная математика сэмплера (выполняется всегда) ---
            
            # Расчет Alpha/Sigma
            t_float_curr = torch.full((n_samples, 1, 1, 1), t_curr, device=device, dtype=torch.float)
            sigma_t = noise_scheduler_fn(t_float_curr, n_steps)
            alpha_t = 1.0 - sigma_t 
            
            if t_prev >= 0:
                t_float_prev = torch.full((n_samples, 1, 1, 1), t_prev, device=device, dtype=torch.float)
                sigma_prev = noise_scheduler_fn(t_float_prev, n_steps)
                alpha_prev = 1.0 - sigma_prev
            else:
                sigma_prev = torch.zeros_like(sigma_t)
                alpha_prev = torch.ones_like(alpha_t)

            # Вычисляем lambda (для истории DPM++)
            # Примечание: функция get_coefficients должна быть доступна в контексте
            lambda_t, _, _ = get_coefficients(alpha_t, sigma_t, alpha_prev, sigma_prev)

            if sampling_method == "ddim":
                eps = (x_gen - alpha_t * model_out) / (sigma_t + 1e-8)
                x_gen = alpha_prev * model_out + sigma_prev * eps

            elif sampling_method == "dpm++":
                current_order = 1 if is_last_step else 2
                
                x_gen = multistep_dpm_solver_update(
                    x_gen, model_out, history_buffer, 
                    alpha_t, sigma_t, alpha_prev, sigma_prev, order=current_order
                )
                history_buffer.append((model_out, lambda_t))

            elif sampling_method == "unipc":
                current_order = 1 
                
                if is_last_step:
                     x_gen = multistep_dpm_solver_update(x_gen, model_out, [], alpha_t, sigma_t, alpha_prev, sigma_prev, order=1)
                else:
                     x_gen = multistep_dpm_solver_update(x_gen, model_out, history_buffer, alpha_t, sigma_t, alpha_prev, sigma_prev, order=2)
                
                history_buffer.append((model_out, lambda_t))
            
            # Очистка истории
            if len(history_buffer) > 2:
                history_buffer.pop(0)

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

        if visualize_test_batch and fixed_test_batch is not None and test_visualization_func is not None:
            model.eval() 
            x_test_real, y_test = fixed_test_batch
            y_test = y_test.to(device) 
          
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
    dataloader: torch.utils.data.DataLoader,
    n_steps: int,
    device: str,
    noise_scheduler_name: str = "cosine",     
    output_path: str = "generated_data.npz",
    sampling_method: str = "ddim",
    cache_interval: int = 1,  # Оставляем для совместимости
    compute_steps_schedule: Optional[List[int]] = None, # <--- НОВЫЙ ПАРАМЕТР
    save_all_steps: bool = False,
    specific_steps: Optional[List[int]] = None
):
    # Проверка наличия scheduler (как в оригинале)
    if 'NOISE_SCHEDULERS' in globals():
         noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    else:
         pass 

    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    all_real_images = []
    all_gen_images = [] 
    all_conditions = []
    
    model.to(device)
    model.eval()

    # Логирование режима работы
    if compute_steps_schedule is not None:
        mode_msg = f"Custom Schedule: {compute_steps_schedule}"
    else:
        mode_msg = f"Interval: {cache_interval}"

    print(f"Start Inference: Steps={n_steps}, Mode={mode_msg}, Save All={save_all_steps}")

    with torch.no_grad(): 
        for x_real, y_cond in tqdm(dataloader, desc="Inference and Saving"):

            # Передаем новый параметр в функцию sample
            # ПРИМЕЧАНИЕ: Функция sample должна быть обновлена, чтобы принимать этот аргумент!
            x_gen = sample(
                model, 
                y_cond, 
                n_steps, 
                device,
                noise_scheduler_fn,              
                shape=x_real.shape[1:],
                sampling_method=sampling_method,
                cache_interval=cache_interval,
                compute_steps_schedule=compute_steps_schedule, # <--- Передаем список дальше
                return_all_steps=save_all_steps,
                specific_steps=specific_steps
            )
            
            all_real_images.append(x_real.cpu().numpy())
            all_gen_images.append(x_gen.cpu().numpy())
            all_conditions.append(y_cond.cpu().numpy())

    real_images_np = np.concatenate(all_real_images, axis=0)
    gen_images_np = np.concatenate(all_gen_images, axis=0) 
    conditions_np = np.concatenate(all_conditions, axis=0)

    # --- Блок сохранения (без изменений) ---
    if save_all_steps:
        final_gen_only = gen_images_np[:, -1, ...] 
        
        np.savez_compressed(
            output_path,
            real_images=real_images_np,
            gen_images_history=gen_images_np,
            gen_images_final=final_gen_only,
            conditions=conditions_np,
            labels=np.zeros(len(gen_images_np))
        )
        print(f"Данные с историей шагов сохранены в: '{output_path}'")
        print(f"Размерность истории: {gen_images_np.shape}")

    else:
        real_labels = np.ones(len(real_images_np))
        gen_labels = np.zeros(len(gen_images_np))

        final_images = np.concatenate([real_images_np, gen_images_np], axis=0)
        final_labels = np.concatenate([real_labels, gen_labels], axis=0)
        final_conditions = np.concatenate([conditions_np, conditions_np], axis=0)

        np.savez_compressed(
            output_path,
            images=final_images,
            labels=final_labels,
            conditions=final_conditions
        )
        print(f"Данные (только финал) сохранены в: '{output_path}'")


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

    valid_real_long_width = metrics["Real Longitudual Width"][np.isfinite(metrics["Real Longitudual Width"])]
    valid_real_trans_width = metrics["Real Transverse Width"][np.isfinite(metrics["Real Transverse Width"])]

    max_real_long = np.max(valid_real_long_width) if len(valid_real_long_width) > 0 else np.inf
    max_real_trans = np.max(valid_real_trans_width) if len(valid_real_trans_width) > 0 else np.inf
    
    max_width_threshold = max(max_real_long, max_real_trans)

    gen_mask = (metrics["Gen Longitudual Width"] <= max_width_threshold) & \
               (metrics["Gen Transverse Width"] <= max_width_threshold) & \
               np.isfinite(metrics["Gen Longitudual Width"]) & \
               np.isfinite(metrics["Gen Transverse Width"])

    real_mask = (metrics["Real Longitudual Width"] <= max_width_threshold) & \
                (metrics["Real Transverse Width"] <= max_width_threshold) & \
                np.isfinite(metrics["Real Longitudual Width"]) & \
                np.isfinite(metrics["Real Transverse Width"])
    
    for key in metrics.keys():
        if key.startswith("Gen"):
            metrics[key] = metrics[key][gen_mask]
        elif key.startswith("Real"):
            metrics[key] = metrics[key][real_mask]

    gen_physics_stats = np.stack([
        metrics["Gen Longitudual Asymmetry"],
        metrics["Gen Transverse Asymmetry"],
        metrics["Gen Longitudual Width"],
        metrics["Gen Transverse Width"]
    ], axis=1)
    
    real_physics_stats = np.stack([
        metrics["Real Longitudual Asymmetry"],
        metrics["Real Transverse Asymmetry"],
        metrics["Real Longitudual Width"],
        metrics["Real Transverse Width"]
    ], axis=1)

    precision_energy, recall_energy = calc_pr_rec_from_embeds(
        gen_images.reshape(gen_images.shape[0], -1), 
        real_images.reshape(real_images.shape[0], -1), 
        num_clusters=num_clusters

    )
    
    precision_physics, recall_physics = calc_pr_rec_from_embeds(
        gen_physics_stats, 
        real_physics_stats, 
        num_clusters=num_clusters,
        enforce_balance=False  
    )

    metrics.update({
        'PRD_energy_AUC': np.trapezoid(precision_energy, recall_energy),
        'precision_energy': precision_energy, 'recall_energy': recall_energy,
        'PRD_physics_AUC': np.trapezoid(precision_physics, recall_physics),
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
    Ось X на гистограммах ограничена диапазоном реальных данных.
    """
    scores = _calculate_physics_metrics(
        gen_images.cpu().numpy(), real_images.cpu().numpy(), conditions.cpu().numpy(), num_clusters
    )

    print(f"--- Результаты Физических Метрик ---\nPRD Energy AUC: {np.mean(scores['PRD_energy_AUC']):.4f}\nPRD Physics AUC: {np.mean(scores['PRD_physics_AUC']):.4f}\n------------------------------------")
    sns.set_theme(style="whitegrid")

    for statistic in statistics_to_plot:
        gen_data = scores['Gen ' + statistic]
        real_data = scores['Real ' + statistic]

        gen_df = pd.DataFrame({'value': gen_data, 'source': 'Generated'})
        real_df = pd.DataFrame({'value': real_data, 'source': 'Real'})
        combined_df = pd.concat([gen_df, real_df])

        min_val = np.min(real_data)
        max_val = min(np.max(real_data), 60)
        padding = (max_val - min_val) * 0.05
        
        x_min_limit = min_val - padding
        x_max_limit = max_val + padding
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=combined_df,
            x='value',
            hue='source',
            bins=100,
            binrange=(x_min_limit, x_max_limit),
            alpha=0.6,
            kde=True,
            palette={'Generated': 'orange', 'Real': 'blue'}
        )
        
        plt.title(f"Distribution of {statistic}", fontsize=14, fontweight='bold')
        plt.xlabel(statistic)
        plt.xlim(x_min_limit, x_max_limit)
        plt.tight_layout()
        plt.show()

    print('Energy PRD Curve')
    plot_pr_aucs(scores['precision_energy'], scores['recall_energy'])
    plt.show()

    print('Physics PRD Curve')
    plot_pr_aucs(scores['precision_physics'], scores['recall_physics'])
    plt.show()
    return scores

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

def evaluate_metrics_with_frozen_guidance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_steps: int,                    # Например, 100
    t_train_max: int,                # Например, 1000
    device: str,
    unet_update_freq: int = 10,      # <--- НОВЫЙ ПАРАМЕТР: обновляем Unet каждые 10 шагов
    denoising_scheduler_name: str = "cosine",
    initial_noise: Optional[torch.Tensor] = None,
    apply_expm1: bool = True
) -> Dict[str, List[float]]:
    
    # ... (инициализация scheduler, x_gen_cpu, real_images_eval как раньше) ...
    noise_scheduler_fn = NOISE_SCHEDULERS.get(denoising_scheduler_name)
    model.to(device)
    model.eval()

    # Подготовка данных (как было у тебя)
    all_x_real = []
    all_y_conditions = []
    for x_b, y_b in dataloader:
        all_x_real.append(x_b)
        all_y_conditions.append(y_b)
    x_real_cpu = torch.cat(all_x_real, dim=0)
    y_conditions_cpu = torch.cat(all_y_conditions, dim=0)
    
    if initial_noise is None:
        x_gen_cpu = torch.randn_like(x_real_cpu)
    else:
        x_gen_cpu = initial_noise.clone()

    batch_size = dataloader.batch_size or len(x_real_cpu)
    n_samples = len(x_real_cpu)

    # Словарь для кэширования предсказаний модели (чтобы использовать их N раз)
    # Ключ: индекс батча (j), Значение: тензор pred_x0
    cached_predictions = {} 

    metrics_history = {
        'step': [], 'timestep': [], 
        'PRD_energy_AUC': [], 'PRD_physics_AUC': [],
        'PRD_energy_AUC_std': [], 'PRD_physics_AUC_std': []
    }

    # Подготовка реальных данных для сравнения
    if apply_expm1:
        real_images_eval = torch.expm1(x_real_cpu)
    else:
        real_images_eval = x_real_cpu
    real_images_np = real_images_eval.cpu().numpy()
    conditions_np = y_conditions_cpu.numpy()

    print(f"Запуск: шагов {n_steps}, обновление Unet каждые {unet_update_freq} шагов.")

    with torch.no_grad():
        # Идем от 100 до 0
        for i in tqdm(reversed(range(n_steps + 1)), desc="Evaluating", total=n_steps + 1):
            
            # Расчет времени t (текущее) и t_prev (следующее)
            t_val = torch.floor(torch.tensor(i) * (t_train_max / n_steps)).long()
            t_prev_val = torch.floor(torch.tensor(i - 1) * (t_train_max / n_steps)).clamp(min=0).long()
            
            # Коэффициенты шума для шедулера
            noise_amount_t = noise_scheduler_fn(t_val.float(), t_train_max).to(device)
            signal_amount_t = 1.0 - noise_amount_t
            noise_amount_t_prev = noise_scheduler_fn(t_prev_val.float(), t_train_max).to(device)
            signal_amount_t_prev = 1.0 - noise_amount_t_prev

            # === ЛОГИКА "ЗАМОРОЗКИ" ===
            # Мы запускаем тяжелый model() только если шаг кратен частоте обновления
            # ИЛИ если это самый первый шаг (чтобы инициализировать кэш)
            should_update_unet = (i % unet_update_freq == 0) or (i == n_steps)

            generated_x0_for_step = []     # Для метрик
            generated_x_prev_for_step = [] # Для следующего шага

            for j in range(0, n_samples, batch_size):
                x_gen_batch = x_gen_cpu[j:j+batch_size].to(device)
                y_conditions_batch = y_conditions_cpu[j:j+batch_size].to(device)
                
                # 1. ПОЛУЧЕНИЕ PRED_X0 (Свежее или из кэша)
                if should_update_unet:
                    t_tensor_batch = torch.full((x_gen_batch.shape[0],), t_val.item(), device=device, dtype=torch.long)
                    # !!! ТЯЖЕЛАЯ ОПЕРАЦИЯ !!!
                    pred_x0_batch = model(x_gen_batch, t_tensor_batch, y_conditions_batch)
                    # Сохраняем в кэш
                    cached_predictions[j] = pred_x0_batch
                else:
                    # !!! БЕРЕМ ИЗ КЭША (бесплатно) !!!
                    # Важно: мы применяем СТАРОЕ предсказание к НОВОМУ зашумленному x_gen_batch
                    pred_x0_batch = cached_predictions[j]

                # 2. ШАГ ШЕДУЛЕРА (DDIM Math)
                # Математика выполняется ВСЕГДА, чтобы физически уменьшить шум в картинке
                s_t_batch = signal_amount_t.view(-1, 1, 1, 1)
                n_t_batch = noise_amount_t.view(-1, 1, 1, 1)
                s_prev_batch = signal_amount_t_prev.view(-1, 1, 1, 1)
                n_prev_batch = noise_amount_t_prev.view(-1, 1, 1, 1)

                # Вычисляем "направление шума" на основе (текущего x) и (предсказанного x0)
                pred_noise_batch = (x_gen_batch - s_t_batch * pred_x0_batch) / (n_t_batch + 1e-8)
                
                # Делаем шаг в сторону чистого изображения
                x_gen_next_batch = s_prev_batch * pred_x0_batch + n_prev_batch * pred_noise_batch
                
                generated_x0_for_step.append(pred_x0_batch.cpu())
                generated_x_prev_for_step.append(x_gen_next_batch.cpu())

            # Обновляем x_gen для следующего цикла
            x_gen_cpu = torch.cat(generated_x_prev_for_step, dim=0)

            # 3. РАСЧЕТ МЕТРИК (На КАЖДОМ шаге)
            # Мы хотим видеть график, поэтому считаем метрики всегда
            pred_x0_cpu_all = torch.cat(generated_x0_for_step, dim=0)
            gen_images_eval = torch.maximum(pred_x0_cpu_all, torch.tensor(0.))
            if apply_expm1:
                gen_images_eval = torch.expm1(gen_images_eval)
            gen_images_np = gen_images_eval.cpu().numpy()

            current_metrics = _calculate_physics_metrics(gen_images_np, real_images_np, conditions_np)
            
            # Логируем
            metrics_history['step'].append(i) 
            metrics_history['timestep'].append(t_val.item())
            
            # ... (твоя логика AUC) ...
            auc_energy, current_prd_auc_energy_std = calculate_pr_metrics(current_metrics['precision_energy'], current_metrics['recall_energy'])
            auc_physics, current_prd_auc_physics_std = calculate_pr_metrics(current_metrics['precision_physics'], current_metrics['recall_physics'])
            
            metrics_history['PRD_energy_AUC'].append(auc_energy)
            metrics_history['PRD_physics_AUC'].append(auc_physics)
            metrics_history['PRD_energy_AUC_std'].append(current_prd_auc_energy_std)
            metrics_history['PRD_physics_AUC_std'].append(current_prd_auc_physics_std)

    print("Анализ по шагам завершен.")
        
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
    
    plt.xlabel(f"Denoising Step (0 -> {n_steps})") 
    plt.ylabel("AUC Value")
    plt.title("Изменение PRD AUC в процессе Denoising'а (Исправлено)")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    try:
        dummy_x = torch.randn(batch_size, channels, image_size, image_size).to(device)
        dummy_y = torch.randn(batch_size, conditions_dim).float().to(device)
        dummy_t = 0  
        thop_inputs = (dummy_x, dummy_t, dummy_y)
    except Exception as e:
        print(f"Ошибка при создании фиктивных тензоров: {e}")
        return -1.0

    total_gflops_per_batch = -1.0 
    try:
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
    return total_gflops_per_batch

def get_teacher_trajectory(
    teacher_model: torch.nn.Module, 
    x_curr: torch.Tensor, 
    t_start: int, 
    t_end: int, 
    K_steps: int, 
    y_cond: torch.Tensor, 
    noise_scheduler_fn: Callable, 
    n_inference_steps: int, 
    device: str
) -> torch.Tensor:
    """
    Прогоняет замороженную модель-учителя на K шагов от t_start до t_end (DDIM).
    """
    # Создаем микро-шаги для учителя
    t_steps = np.linspace(t_start, t_end, K_steps + 1).astype(int)
    x_gen = x_curr.clone()
    
    with torch.no_grad():
        for i in range(K_steps):
            t_current = t_steps[i]
            t_next = t_steps[i+1]
            
            t_tensor = torch.full((x_gen.shape[0],), t_current, device=device, dtype=torch.long)
            model_out = teacher_model(x_gen, t_tensor, y_cond)
            
            t_float_curr = torch.full((x_gen.shape[0], 1, 1, 1), t_current, device=device, dtype=torch.float)
            sigma_t = noise_scheduler_fn(t_float_curr, n_inference_steps)
            alpha_t = 1.0 - sigma_t
            
            t_float_next = torch.full((x_gen.shape[0], 1, 1, 1), t_next, device=device, dtype=torch.float)
            sigma_next = noise_scheduler_fn(t_float_next, n_inference_steps)
            alpha_next = 1.0 - sigma_next
            
            # Стандартный шаг DDIM
            eps = (x_gen - alpha_t * model_out) / (sigma_t + 1e-8)
            x_gen = alpha_next * model_out + sigma_next * eps
            
    return x_gen # Это x_{end}, к которому должен прийти студент


def train_distillation(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    schedule: List[int], # <--- Тот самый "опорный лист" (например, от get_power_schedule)
    teacher_steps_per_interval: int = 2, # Сколько шагов делает учитель внутри интервала (1-2 -> 1)
    noise_scheduler_name: str = "cosine",
    n_inference_steps: int = 1000,
    checkpoint_path: str = "./distilled_checkpoints",
) -> Dict[str, List[float]]:
    """
    Функция прогрессивной дистилляции модели по заданному расписанию (опорному листу).
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    teacher_model.eval() # Учитель заморожен
    
    history = {'train_loss': []}
    best_train_loss = float('inf')

    # Убеждаемся, что расписание отсортировано по убыванию (от T к 0)
    schedule = sorted(schedule, reverse=True)

    for epoch in range(n_epochs):
        print(f"--- Distillation Epoch {epoch + 1}/{n_epochs} ---")
        student_model.train()
        epoch_train_loss = []
        
        for x, y in tqdm(train_loader, desc="Distilling"):
            x, y = x.to(device), y.to(device)
            bs = x.shape[0]
            
            # 1. Случайно выбираем интервал из нашего опорного листа
            # idx от 0 до len(schedule) - 2
            interval_indices = torch.randint(0, len(schedule) - 1, (bs,))
            
            t_start_vals = torch.tensor([schedule[i] for i in interval_indices], device=device)
            t_end_vals = torch.tensor([schedule[i+1] for i in interval_indices], device=device)
            
            # Для простоты батчевых вычислений, возьмем один интервал на весь батч
            # (если нужна побатчевая рандомизация t, потребуется чуть более сложный сбор t_steps)
            t_start = t_start_vals[0].item()
            t_end = t_end_vals[0].item()

            # 2. Зашумляем реальные данные до t_start
            t_start_float = torch.full((bs, 1, 1, 1), t_start, device=device, dtype=torch.float)
            sigma_start = noise_scheduler_fn(t_start_float, n_inference_steps)
            alpha_start = 1.0 - sigma_start
            
            noise = torch.randn_like(x)
            x_start = x * alpha_start + noise * sigma_start
            
            # 3. Учитель делает K шагов, чтобы получить x_{end}
            x_end_target = get_teacher_trajectory(
                teacher_model, x_start, t_start, t_end, 
                teacher_steps_per_interval, y, 
                noise_scheduler_fn, n_inference_steps, device
            )
            
            # 4. Вычисляем идеальный target_x0 для Студента
            t_end_float = torch.full((bs, 1, 1, 1), t_end, device=device, dtype=torch.float)
            sigma_end = noise_scheduler_fn(t_end_float, n_inference_steps)
            alpha_end = 1.0 - sigma_end
            
            c = sigma_end / (sigma_start + 1e-8)
            numerator = x_end_target - c * x_start
            denominator = alpha_end - c * alpha_start + 1e-8
            target_x0 = numerator / denominator
            
            # 5. Обучаем Студента предсказывать этот target_x0
            t_tensor_student = torch.full((bs,), t_start, device=device, dtype=torch.long)
            student_pred = student_model(x_start, t_tensor_student, y)
            
            # Используем твой лосс (обычно MSE или Huber)
            loss = loss_fn(target_x0, student_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())

        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        history['train_loss'].append(avg_train_loss)
        print(f"Avg Distillation Loss: {avg_train_loss:.5f}")
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(student_model.state_dict(), os.path.join(checkpoint_path, "best_distilled_model.pth"))
            print(f"🚀 New best distilled model saved!")

    print("Distillation finished.")
    return history
