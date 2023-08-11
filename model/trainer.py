
#from flax.training import checkpoints
import json
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
import equinox as eqx

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
        self.model = self.model_class(**self.model_hparams, key = model_key)
        # Prepare logging
        self.log_dir = os.path.join(path, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.path = path
        self.losstype = losstype
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model


    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        @eqx.filter_value_and_grad
        def calculate_loss_acc(model, ts, batch):
            y, args = batch
            batch = (y[:, 0], args)
            y_pred = jax.vmap(model, in_axes=(None, 0))(ts, batch)
            if self.losstype=="sin":
                loss = jnp.mean((jnp.sin(y) - jnp.sin(y_pred)) ** 2)
            elif self.losstype == "criterion":
                loss = jnp.mean((y - y_pred) ** 2)
                print(f'y_pred shape is {y_pred.shape}')
            return loss

        # Training function
        # Jit the function for efficiency
        @eqx.filter_jit
        def train_step(ts, batch, model, opt_state):
            loss, grads = calculate_loss_acc(model, ts, batch)
            print(f'grads are {grads}')
            updates, opt_state = self.optimizer.update(grads, opt_state)
            print(f'opt_state is {opt_state}')
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state


        # Eval function
        # Jit the function for efficiency
        @eqx.filter_jit
        def eval_step(ts, batch, model):
            # Determine the accuracy
            y, args = batch
            batch = (y[:, 0], args)
            y_pred = jax.vmap(model, in_axes=(None, 0))(ts, batch)
            if self.losstype == "sin":
                loss = jnp.mean((jnp.sin(y) - jnp.sin(y_pred)) ** 2)
            elif self.losstype == "criterion":
                loss = jnp.mean((y - y_pred) ** 2)
            return loss

        self.train_step = train_step
        self.eval_step = eval_step


    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'adablf':
            opt_class = optax.adabelief
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
            {int(num_steps_per_epoch * num_epochs * 0.6): 0.05,
             int(num_steps_per_epoch * num_epochs * 0.85): 0.01}
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        self.optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )
        #self.optimizer = opt_class(self.optimizer_hparams.pop('lr'))

        # Initialize training state
        #print(f'self.optimizer is {self.optimizer}')
        print(f' model is {eqx.filter(self.model, eqx.is_inexact_array)}')
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))
        print(f'self.opt_state is {self.opt_state}')

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
                    hyperparams = {'epoch': epoch_idx,
                                   'ratio': ratio}
                    self.save_model(os.path.join(self.path,
                                                 f'{self.model_name}.ckpt'),
                                    hyperparams)
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
            loss, self.model, self.opt_state = self.train_step(ti, (yi, args), self.model, self.opt_state)
            metrics['loss'].append(loss)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/' + key, avg_val, global_step=epoch)
        return loss

    def eval_model(self, ts, data_loader):
        # Test model on all images of a data loader and return avg loss
        loss_val, count = 0, 0
        inference_model = eqx.tree_inference(self.model, value=True)
        for batch in data_loader:
            y, b, adj = batch
            loss = self.eval_step(ts, (y, (b, adj)), inference_model)
            loss_val += loss * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_loss = (loss_val / count).item()
        return eval_loss

    def save_model(self, filename, hyperparams): #hyperparams = step, epoch, ratio
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

    def load_model(self):
        print(f'wer are here to draw')
        with open(os.path.join(self.path, f'{self.model_name}.ckpt'), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = self.model_class(**self.model_hparams, key = self.model_key)
            self.model = eqx.tree_deserialise_leaves(f, model)
        return hyperparams['epoch'], hyperparams['ratio']


    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.exists(os.path.join(self.path, f'{self.model_name}.ckpt'))