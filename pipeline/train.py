import contextlib
from typing import Tuple, Generator, Dict, Any, Optional, ContextManager, Callable, List
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm

from pipeline.aux import calc_grad_norm
from pipeline.data import collate_fn, move_batch_to, get_random_infinite_dataloader
from pipeline.device import get_local_device
from pipeline.gan import GAN
from pipeline.logger import GANLogger
from pipeline.metrics import Metric, MetricsSequence
from pipeline.metrics_logging import log_metric
from pipeline.normalization import update_normalizers_stats
from pipeline.predicates import TrainPredicate
from pipeline.regularizer import Regularizer
from pipeline.results_storage import ExperimentInfo, Result
from pipeline.storage import ModelDir


class Stepper:
    """обёртка над всем необходимым для шага градиентного спуска"""
    def __init__(self, optimizer: optim.Optimizer,
                 scheduler=None, scheduler_mode: str = 'epoch') -> None:
        """
        TODO: над scheduler хочу сделать обёртку, чтобы передавать в него вообще всё во время обучение

        scheduler_mode - mode of the scheduler
           'epoch' - step after each epoch
           'batch' - step after each batch
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_mode = scheduler_mode

    def step(self, *args, **kwargs) -> None:
        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_mode == 'batch':
            self.scheduler.step(*args, **kwargs)

    def epoch_finished(self, *args, **kwargs) -> None:
        if self.scheduler is not None and self.scheduler_mode == 'epoch':
            self.scheduler.step(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])


# знает, как обучать GAN
class GanEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None,
                    regularizer: Optional[Regularizer] = None) -> None:
        pass


def check_tensor(x: torch.Tensor, prefix: str = ''):
    msg = prefix
    if x.isnan().any():
        msg += 'NaNs'
    elif x.isinf().any():
        msg += '+infs'
    elif x.isneginf().any():
        msg += '-infs'
    else:
        return
    raise ValueError(msg)


class WganEpochTrainer(GanEpochTrainer):
    def __init__(self, n_critic: int = 5, batch_size: int = 64) -> None:
        self.n_critic = n_critic
        self.batch_size = batch_size
        # probably, these counters should be moved to logger
        self.gen_batch_cnt = 0
        self.disc_batch_cnt = 0

    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None,
                    regularizer: Optional[Regularizer] = None) -> None:
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        random_dataloader = get_random_infinite_dataloader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        random_dataloader_iter = iter(random_dataloader)

        def get_batches(real_batch) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
            """Look at the return statement"""
            real_batch_x, real_batch_y = move_batch_to(real_batch, get_local_device())
            gen_batch_y = real_batch_y
            noise_batch_z = gan_model.gen_noise(len(real_batch_x)).to(get_local_device())
            gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())

            return gen_batch_x, real_batch_x, gen_batch_y, real_batch_y

        critic_adv_loss_total = 0.
        critic_loss_total = 0.
        gen_adv_loss_total = 0.
        gen_loss_total = 0.
        regularizer_loss_total = 0.
        disc_grad_norm_total = 0.
        gen_grad_norm_total = 0.
        
        # generator_batch = next(random_dataloader_iter)  # for local testing
        # for batch_index in range(1):                    # -----------------
        for batch_index, generator_batch in enumerate(tqdm(dataloader)):
            # critic training
            gan_model.generator.requires_grad_(False)
            for t in range(self.n_critic):
                self.disc_batch_cnt += 1
                real_batch = next(random_dataloader_iter)
                gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(real_batch)
                check_tensor(gen_batch_x, 'Generated values contain ')
                disc_real_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                check_tensor(disc_real_vals, 'Discriminator values for real data contain ')
                disc_gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                check_tensor(disc_gen_vals, 'Discriminator values for generated data contain ')
                loss = - (disc_real_vals - disc_gen_vals).mean()
                critic_adv_loss_total += loss.item() * len(gen_batch_x)
                if regularizer is not None:
                    regularizer_loss = regularizer()
                    regularizer_loss_total += regularizer_loss.item()
                    if logger is not None:
                        logger.log_metrics(data={'train/discriminator/pure_loss': loss.item(),
                                                 'train/discriminator/reg_loss': regularizer_loss.item()},
                                           period='disc_batch', period_index=self.disc_batch_cnt, commit=False)
                    loss += regularizer_loss
                loss.backward()
                disc_grad_norm = calc_grad_norm(gan_model.discriminator)
                disc_grad_norm_total += disc_grad_norm
                if logger is not None:
                    logger.log_metrics(data={'train/discriminator/batch_loss': loss.item(),
                                             'train/discriminator/batch_grad_norm': disc_grad_norm},
                                       period='disc_batch', period_index=self.disc_batch_cnt, commit=True)
                critic_stepper.step()
                critic_stepper.optimizer.zero_grad()
                update_normalizers_stats(gan_model.discriminator, disc_real_vals=disc_real_vals,
                                         disc_gen_vals=disc_gen_vals)

            critic_loss_total += loss.item() * len(gen_batch_x)  # intentionally not in 'for' loop

            gan_model.generator.requires_grad_(True)

            # generator training
            gan_model.discriminator.requires_grad_(False)
            gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(generator_batch)

            observations = (gan_model.discriminator(real_batch_x, real_batch_y) -
                            gan_model.discriminator(gen_batch_x, gen_batch_y))
            gen_loss = observations.mean()
            self.gen_batch_cnt += 1
            gen_adv_loss_total += gen_loss.item() * len(gen_batch_x)
            if regularizer is not None:
                regularizer_loss = regularizer()
                regularizer_loss_total += regularizer_loss.item()
                if logger is not None:
                    logger.log_metrics(data={'train/generator/pure_loss': gen_loss.item(),
                                             'train/generator/reg_loss': regularizer_loss.item()},
                                       period='gen_batch', period_index=self.gen_batch_cnt, commit=False)
                gen_loss += regularizer_loss
                regularizer.step()
            gen_loss_total += gen_loss * len(gen_batch_x)
            gen_loss.backward()
            gen_grad_norm = calc_grad_norm(gan_model.generator)
            gen_grad_norm_total += gen_grad_norm
            generator_stepper.step()
            generator_stepper.optimizer.zero_grad()
            gan_model.discriminator.requires_grad_(True)

            if logger is not None:
                logger.log_metrics(data={'train/generator/batch_loss': gen_loss.item(),
                                         'train/generator/batch_grad_norm': gen_grad_norm},
                                   period='gen_batch', period_index=self.gen_batch_cnt, commit=True)

        generator_stepper.epoch_finished()
        critic_stepper.epoch_finished()

        if logger is not None:
            if generator_stepper.scheduler is not None:
                generator_lr = generator_stepper.scheduler.get_last_lr()
                logger.log_metrics(
                    data={'generator/lr': generator_lr},
                    period='epoch',
                    commit=False)
            if critic_stepper.scheduler is not None:
                critic_lr = critic_stepper.scheduler.get_last_lr()
                logger.log_metrics(
                    data={'critic/lr': critic_lr},
                    period='epoch',
                    commit=False)

            logger.log_metrics(data={'train/critic/loss': critic_loss_total / len(train_dataset),
                                     'train/critic/adv_loss': critic_adv_loss_total / len(train_dataset),
                                     'train/generator/loss': gen_loss_total / len(train_dataset),
                                     'train/generator/adv_loss': gen_adv_loss_total / len(train_dataset),
                                     'train/regularizer/loss': regularizer_loss_total / (len(dataloader) * (self.n_critic + 1)),
                                     'train/discriminator/grad_norm': disc_grad_norm_total / len(dataloader) * self.n_critic,
                                     'train/generator/grad_norm': gen_grad_norm_total / len(dataloader)},
                               period='epoch',
                               commit=False)  # period_index was specified by the caller


def fill_result(result: Result, metric_names: List, metric_values: List):
    for metric_name, metric_value in zip(metric_names, metric_values):
        if isinstance(metric_name, List):
            fill_result(result, metric_name, metric_value)
        else:
            result.add_metric(metric_name, value=metric_value)


# знает, что нужно для обучения GAN
class GanTrainer:
    def __init__(self, model_dir: ModelDir, save_checkpoint_once_in_epoch: int = 1,
                 use_saved_checkpoint: bool = True) -> None:
        self.model_dir = model_dir
        self.save_checkpoint_once_in_epoch = save_checkpoint_once_in_epoch
        self.use_saved_checkpoint = use_saved_checkpoint

    def train(self, gan_model: GAN,
              train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
              generator_stepper: Stepper, critic_stepper: Stepper,
              epoch_trainer: GanEpochTrainer,
              n_epochs: int = 100,
              metric: Optional[Metric] = None,
              metric_predicate: Optional[TrainPredicate] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[GANLogger]]] = None,
              regularizer: Optional[Regularizer] = None,
              normalization_loss: Optional[Callable] = None,
              result_metrics: Optional[Tuple[List, MetricsSequence]] = None,
              results_info: Optional[ExperimentInfo] = None) -> Generator[Tuple[int, GAN], None, GAN]:
        """
        :param train_dataset: is expected to return tuples (x, y)
        :param val_dataset: is expected to return tuples (x, y)
        :param metric: metric that will be calculated and logged (only if logger is given) after each epoch
        :param metric_predicate: `metric` is calculated if and only if `metric_predicate` returns True
        :return:
        """
        gan_model.to(get_local_device())
        inverse_to_initial_domain_fn = getattr(train_dataset, 'inverse_transform', None)
        epoch = 1

        if self.use_saved_checkpoint:
            checkpoint = self.model_dir.get_checkpoint_state()
            if checkpoint is not None:
                epoch = checkpoint['epoch']
                print(f"Checkpoint was loaded. Current epoch: {epoch}")
                gan_model.load_state_dict(checkpoint['gan'])
                generator_stepper.load_state_dict(checkpoint['generator_stepper'])
                critic_stepper.load_state_dict(checkpoint['critic_stepper'])

        if logger_cm_fn is None:
            logger_cm = contextlib.nullcontext(None)
        else:
            logger_cm = logger_cm_fn() or contextlib.nullcontext(None)

        with logger_cm as logger:
            while epoch <= n_epochs:
                if logger is not None:
                    logger.log_metrics(data={}, period='epoch', period_index=epoch, commit=False)

                epoch_trainer.train_epoch(gan_model=gan_model, train_dataset=train_dataset, val_dataset=val_dataset,
                                          generator_stepper=generator_stepper, critic_stepper=critic_stepper,
                                          logger=logger, regularizer=regularizer)

                if logger is not None:
                    if metric is not None and (metric_predicate is None or metric_predicate(epoch=epoch)):
                        with torch.no_grad():
                            metrics_results = metric(gan_model=gan_model, train_dataset=train_dataset, val_dataset=val_dataset,
                                                     inverse_to_initial_domain_fn=inverse_to_initial_domain_fn)
                        log_metric(metric, results=metrics_results, logger=logger, period='epoch', period_index=epoch)
                    logger.commit(period='epoch')
                epoch += 1

                if self.save_checkpoint_once_in_epoch != 0 and epoch % self.save_checkpoint_once_in_epoch == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'gan': gan_model.state_dict(),
                        'generator_stepper': generator_stepper.state_dict(),
                        'critic_stepper': critic_stepper.state_dict()
                    }
                    self.model_dir.save_checkpoint_state(checkpoint)

                yield epoch, gan_model

        # evaluating results
        # TODO: probably this logic should be placed somewhere else
        if result_metrics is not None and results_info is not None:
            result = results_info.get_result()
            # metrics
            metrics_values = result_metrics[1](gan_model=gan_model, train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               inverse_to_initial_domain_fn=inverse_to_initial_domain_fn)
            print(metrics_values)
            fill_result(result, result_metrics[0], metrics_values)
            results_info.save_result(result)

        return gan_model
