
from flax.training import checkpoints

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
jax.config.update("jax_enable_x64", True)

## Standard libraries
import os
from typing import Any
from collections import defaultdict

## Imports for plotting
import matplotlib

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
sns.reset_orig()

## Progress bar
from tqdm.auto import tqdm
import optax

from torch.utils.tensorboard import SummaryWriter


class TrainerModule:

    def __init__(self,
                 model_name: str,
                 model_class: nn.Module,  # NeuralODE
                 model_hparams: dict,  # {node_size = node_size, act_fn  = act_fn_by_name['sine']()}
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 ts: Any,
                 exmp_imgs: Any,  # (ys[:,0],bias, data_adj)
                 model_key : int,
                 path : str,
                 losstype : str):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.model_key = model_key
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        # Prepare logging
        self.log_dir = os.path.join(path, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.path = path
        self.losstype = losstype
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(ts, exmp_imgs)


    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss_acc(state, params, ts, batch):
            y, args = batch
            batch = (y[:, 0], args)
            # Obtain the logits and predictions of the model for the input data

            y_pred = jax.vmap(state.apply_fn, in_axes=(None, None, 0))(params, ts, batch)
            if self.losstype=="sin":
                loss = jnp.mean((jnp.sin(y) - jnp.sin(y_pred)) ** 2)
            elif self.losstype == "criterion":
                loss = jnp.mean((y - y_pred) ** 2)
            return loss

        # Training function
        # Jit the function for efficiency
        def train_step(state, ts, batch):
            # Gradient function
            grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                         argnums=1,  # Parameters are second argument of the function
                                         has_aux=False  # Function has additional outputs, here accuracy
                                         )
            # Determine gradients for current model, parameters and batch
            loss, grads = grad_fn(state, state.params, ts, batch)
            # Perform parameter update with gradients and optimizer
            state = state.apply_gradients(grads=grads)
            # Return state and any other value we might want
            return state, loss

        # Eval function
        # Jit the function for efficiency
        def eval_step(state, ts, batch):
            # Determine the accuracy
            loss = calculate_loss_acc(state, state.params, ts, batch)
            return loss

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, ts, exmp_imgs):
        # Initialize model
        variables = self.model.init(self.model_key, ts, exmp_imgs)
        self.init_params = variables
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
            {int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
             int(num_steps_per_epoch * num_epochs * 0.85): 0.1}
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )

        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                   params=self.init_params if self.state is None else self.state.params,
                                                   tx=optimizer)

    def train_model(self, train_loader, eval_loader, ts, num_epochs=100,
                    ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        time_length = len(ts)
        self.init_optimizer(num_epochs, len(train_loader))
        epoch_num = 1
        if self.checkpoint_exists():  # continue training if pretrained model exists
            epoch_num, ratio = self.load_model()
            rr = []
            for r in ratios:
                if r >= ratio:
                    rr.append(r)
            ratios = rr

        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs

        # Track best eval accuracy
        best_eval = 0.0
        for ratio in ratios:
            print(f'ratio is', ratio)
            for epoch_idx in tqdm(range(epoch_num, num_epochs + 1)):
                loss = self.train_epoch(train_loader, ts, time_length, ratio, epoch=epoch_idx)
                if epoch_idx % 100 == 0:
                    eval_loss = self.eval_model(ts, eval_loader)
                    print(f'train_loss is {loss}, eval loss is {eval_loss} and epoch is {epoch_idx}')
                    self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                    self.save_model(step=1, epoch=epoch_idx, ratio=ratio)
                    if eval_loss <= best_eval:
                        best_eval = eval_loss

                    self.logger.flush()
            epoch_num = 1

    def train_epoch(self, train_loader, ts, time_length, ratio, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc='Training', leave=False):
            y, b, adj = batch
            args = (b, adj)
            ti = ts[:int(time_length * ratio)]
            yi = y[:, :int(time_length * ratio)]
            self.state, loss = self.train_step(self.state, ti, (yi, args))
            metrics['loss'].append(loss)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/' + key, avg_val, global_step=epoch)
        return loss

    def eval_model(self, ts, data_loader):
        # Test model on all images of a data loader and return avg loss
        loss_val, count = 0, 0
        for batch in data_loader:
            y,b, adj = batch
            loss = self.eval_step(self.state, ts, (y, (b, adj)))
            loss_val += loss * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_loss = (loss_val / count).item()
        return eval_loss

    def save_model(self, step, epoch, ratio):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=os.path.join(self.path, f'{self.model_name}.ckpt'),
                                    target={'params': self.state.params,
                                            'epoch': epoch,
                                            'ratio': ratio},
                                    step=step,
                                    prefix='my_model',
                                    overwrite=True)

    def load_model(self):
        # Load model. We use different checkpoint for pretrained models
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.path, f'{self.model_name}.ckpt'),
                                                    prefix='my_model',
                                                    target=None)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                   params=state_dict['params'],
                                                   tx=self.state.tx if self.state else optax.sgd(0.1)
                                                   # Default optimizer
                                                   )
        return state_dict['epoch'], state_dict['ratio']

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.exists(os.path.join(self.path, f'{self.model_name}.ckpt'))