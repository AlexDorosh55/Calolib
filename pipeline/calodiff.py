# calodiff.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import copy
from typing import Callable, Optional, Dict, List


# --- Вспомогательные функции (Scheduler'ы шума) ---

def _cosine_noise_scheduler(t: torch.Tensor, t_max: int) -> torch.Tensor:
    """
    Косинусный scheduler для уровня шума.
    Плавное изменение от 0 до 1.
    """
    return 0.5 * (1 - torch.cos(torch.pi * t / t_max))


def _linear_noise_scheduler(t: torch.Tensor, t_max: int) -> torch.Tensor:
    """
    Линейный scheduler для уровня шума.
    Изменение от 0 до 1.
    """
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
        denoising_scheduler_name: str = "cosine"
) -> torch.Tensor:
    """
    Функция для генерации изображений (инференса) с использованием обученной модели.

    Args:
        model: Обученная диффузионная модель.
        y_conditions: Тензор с условиями (метками) для генерации.
        n_steps: Количество шагов в процессе генерации.
        device: Устройство ('cpu' или 'cuda').
        shape: Размерность генерируемого изображения (без учета батча).
        denoising_scheduler_name: Название scheduler'а для процесса генерации.

    Returns:
        Тензор с сгенерированными изображениями.
    """
    n_samples = y_conditions.shape[0]
    x_gen = torch.rand(n_samples, *shape).to(device)  # Начинаем с чистого шума
    y_conditions = y_conditions.to(device)

    denoising_scheduler = NOISE_SCHEDULERS.get(denoising_scheduler_name)
    if not denoising_scheduler:
        raise ValueError(f"Неизвестный scheduler: {denoising_scheduler_name}")

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_steps), desc="Sampling", leave=False):
            t = torch.tensor(i, device=device)
            # В вашем коде timestep всегда 0, я сохранил эту логику.
            # В классических диффузионках сюда передается сам timestep.
            pred = model(x_gen, 0, y_conditions)

            # Логика смешивания из вашего кода
            mix_factor = 1 / (n_steps - i)
            x_gen = x_gen * (1 - mix_factor) + pred * mix_factor

    return x_gen.cpu()


# --- Основная функция обучения ---

def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        n_epochs: int,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: str,
        # --- Параметры Scheduler'ов ---
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        noise_scheduler_name: str = "cosine",
        # --- Параметры валидации и сохранения ---
        validation_freq: int = 1,
        n_inference_steps: int = 1000,
        metric_calculator: Optional[object] = None,
        checkpoint_path: str = "./checkpoints",
        # --- Параметры для ранней остановки (Early Stopping) ---
        early_stopping_patience: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Универсальная функция для обучения диффузионной модели.

    Args:
        model: Модель для обучения.
        train_loader: DataLoader для обучающей выборки.
        valid_loader: DataLoader для валидационной выборки.
        n_epochs: Количество эпох обучения.
        loss_fn: Функция потерь.
        optimizer: Оптимизатор.
        device: Устройство для вычислений ('cpu' или 'cuda').
        lr_scheduler: Scheduler для изменения learning rate.
        noise_scheduler_name: Название scheduler'а для зашумления данных ('cosine' или 'linear').
        validation_freq: Частота проведения валидации (каждые N эпох).
        n_inference_steps: Количество шагов при генерации на этапе валидации.
        metric_calculator: Объект для подсчета и визуализации метрик.
        checkpoint_path: Путь для сохранения моделей.
        early_stopping_patience: Количество эпох без улучшения на валидации для остановки.

    Returns:
        Словарь с историей лоссов.
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    history = {'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1}/{n_epochs} ---")

        # --- Фаза обучения ---
        model.train()
        epoch_train_loss = []
        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)

            # Генерация и применение шума
            t = torch.randint(0, n_inference_steps, (x.shape[0],), device=device)
            noise_amount = noise_scheduler_fn(t, n_inference_steps).view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            noisy_x = x * (1 - noise_amount) + noise * noise_amount

            # Предсказание и расчет потерь
            pred = model(noisy_x, 0, y)  # timestep = 0, как в вашем коде
            loss = loss_fn(x, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())

        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        history['train_loss'].append(avg_train_loss)
        print(f"Avg Train Loss: {avg_train_loss:.5f}")

        # --- Фаза валидации ---
        if (epoch + 1) % validation_freq == 0:
            model.eval()
            epoch_valid_loss = []
            all_predictions = []
            real_images_for_metrics = []
            conditions_for_metrics = []

            with torch.no_grad():
                for x_val, y_val in tqdm(valid_loader, desc="Validation"):
                    # Генерируем изображения для текущих условий y_val
                    generated_images = sample(
                        model, y_val, n_inference_steps, device,
                        shape=(x_val.shape[1], x_val.shape[2], x_val.shape[3])
                    )

                    # Считаем loss для сгенерированных и реальных
                    loss = loss_fn(x_val, generated_images)
                    epoch_valid_loss.append(loss.item())

                    # Сохраняем результаты для метрик
                    all_predictions.append(generated_images)
                    real_images_for_metrics.append(x_val.cpu())
                    conditions_for_metrics.append(y_val.cpu())

            avg_valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
            history['valid_loss'].append(avg_valid_loss)
            print(f"Avg Validation Loss: {avg_valid_loss:.5f}")

            # --- Подсчет метрик ---
            if metric_calculator:
                print("Calculating and visualizing metrics...")
                all_predictions = torch.cat(all_predictions, dim=0)
                real_images_for_metrics = torch.cat(real_images_for_metrics, dim=0)
                conditions_for_metrics = torch.cat(conditions_for_metrics, dim=0)

                metric_calculator.visualize_metrics(
                    all_predictions,
                    real_images_for_metrics,
                    conditions=conditions_for_metrics
                )

            # --- Обновление LR Scheduler ---
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(avg_valid_loss)
                else:
                    lr_scheduler.step()

            # --- Логика сохранения и ранней остановки ---
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, os.path.join(checkpoint_path, "best_model.pth"))
                print(f"✨ New best model saved with validation loss: {best_valid_loss:.5f}")
            elif early_stopping_patience:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print("Stopping early due to no improvement.")
                    model.load_state_dict(best_model_state)  # Загружаем лучшую модель
                    return history

    model.load_state_dict(best_model_state)  # Загружаем лучшую модель в конце
    return history