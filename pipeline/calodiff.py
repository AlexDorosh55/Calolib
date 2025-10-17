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
    n_epochs: int,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    # --- Опциональные параметры ---
    valid_loader: Optional[DataLoader] = None,
    # --- Параметры Scheduler'ов ---
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    noise_scheduler_name: str = "cosine",
    # --- Параметры валидации и сохранения ---
    validation_freq: int = 1,
    n_inference_steps: int = 1000,
    metric_calculator: Optional[object] = None, # Параметр оставлен для совместимости
    checkpoint_path: str = "./checkpoints",
    # --- Параметры для ранней остановки (Early Stopping) ---
    early_stopping_patience: Optional[int] = None,
    # --- ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ НА ТЕСТЕ ---
    test_loader: Optional[DataLoader] = None,
    visualize_test_batch: bool = True,
    test_visualization_func: Optional[Callable] = None
) -> Dict[str, List[float]]:
    """
    Универсальная функция для обучения диффузионной модели.
    Сохраняет лучшую модель на основе улучшения train_loss.
    Валидация и ранняя остановка являются опциональными.
    Визуализация на тестовом батче происходит на каждой эпохе.
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    noise_scheduler_fn = NOISE_SCHEDULERS.get(noise_scheduler_name)
    if not noise_scheduler_fn:
        raise ValueError(f"Неизвестный scheduler шума: {noise_scheduler_name}")

    history = {'train_loss': [], 'valid_loss': []}
    
    # --- Переменные для отслеживания лучших состояний ---
    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_model_state_on_train = None # Состояние модели с лучшим train_loss
    
    patience_counter = 0

    # Заранее фиксируем один батч из test_loader для консистентной визуализации
    fixed_test_batch = None
    if test_loader and visualize_test_batch:
        try:
            fixed_test_batch = next(iter(test_loader))
        except StopIteration:
            print("Warning: test_loader пуст, визуализация на тестовом батче будет пропущена.")

    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1}/{n_epochs} ---")
        
        # --- Фаза обучения ---
        model.train()
        epoch_train_loss = []
        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, n_inference_steps, (x.shape[0],), device=device)
            noise_amount = noise_scheduler_fn(t.float(), n_inference_steps).view(-1, 1, 1, 1).to(device)
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

        # --- Логика сохранения модели по train_loss ---
        # Происходит на каждой эпохе, независимо от валидации
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_model_state_on_train = copy.deepcopy(model.state_dict())
            torch.save(best_model_state_on_train, os.path.join(checkpoint_path, "best_model_on_train.pth"))
            print(f"🚀 New best model saved with train loss: {best_train_loss:.5f}")

        # --- Визуализация на тестовом батче ---
        # Происходит на каждой эпохе, если параметры предоставлены
        if visualize_test_batch and fixed_test_batch is not None and test_visualization_func is not None:
            print("Visualizing examples from the test batch...")
            x_test_real, y_test = fixed_test_batch
            generated_images = sample(
                model, y_test.to(device), n_inference_steps, device,
                shape=(x_test_real.shape[1], x_test_real.shape[2], x_test_real.shape[3])
            )

            n_samples_to_show = min(len(generated_images), 5)
            fig, axs = plt.subplots(1, n_samples_to_show, figsize=(20, 4))
            fig.suptitle(f"Test Batch Visualization at Epoch {epoch + 1}", fontsize=16)

            if n_samples_to_show == 1: axs = [axs] 

            for i, ax in enumerate(axs):
                test_visualization_func(energy=generated_images[i].cpu(), ax=ax)
            plt.show()

        # --- Опциональная фаза валидации ---
        if valid_loader and (epoch + 1) % validation_freq == 0:
            model.eval()
            epoch_valid_loss = []
            with torch.no_grad():
                for x_val, y_val in tqdm(valid_loader, desc="Validation"):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred_val = model(x_val, 0, y_val) # y_val уже на device
                    loss = loss_fn(x_val, pred_val)
                    epoch_valid_loss.append(loss.item())
            
            avg_valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
            history['valid_loss'].append(avg_valid_loss)
            print(f"Avg Validation Loss: {avg_valid_loss:.5f}")
            
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(avg_valid_loss)
                else:
                    lr_scheduler.step()
            
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
                # Можно дополнительно сохранять и лучшую по валидации модель, если нужно
                # best_model_state_on_valid = copy.deepcopy(model.state_dict())
                # torch.save(best_model_state_on_valid, os.path.join(checkpoint_path, "best_model_on_valid.pth"))
                print(f"✨ Validation loss improved to: {best_valid_loss:.5f}")
            elif early_stopping_patience:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Stopping early. No improvement in validation loss for {patience_counter} epochs.")
                    if best_model_state_on_train:
                        model.load_state_dict(best_model_state_on_train)
                    return history

    # В конце обучения загружаем в модель состояние с лучшим train_loss
    print("Training finished. Loading the best model based on training loss.")
    if best_model_state_on_train:
        model.load_state_dict(best_model_state_on_train)
        
    return history
