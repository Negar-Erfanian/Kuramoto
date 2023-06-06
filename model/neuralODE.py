import diffrax
import jax
import jax.numpy as jnp
import flax.linen as nn
jax.config.update("jax_enable_x64", True)

from typing import Callable


## Imports for plotting
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
sns.reset_orig()


class NeuralODE(nn.Module):  # Here I will solve using the solver
    act_fn: Callable
    node_size: int
    normal: str
    hdims : str

    @nn.compact
    def __call__(self, ts, y):
        if not self.normal:
            kernels = self.param('kernel', nn.initializers.normal(), [self.node_size, self.node_size, self.node_size])
        else:
            hdims_list = [int(i) for i in self.hdims.split('-')]
            kernels = []
            first_dim = self.node_size
            for i, dim in enumerate(hdims_list):
                kernels.append(self.param(f'kernel{i}', nn.initializers.normal(), [first_dim, dim]))
                first_dim = dim
            kernels.append(self.param('kernel', nn.initializers.normal(), [first_dim, self.node_size]))
        def fn(t, y, args):
            y0, bias, data_adj = y

            if not self.normal:
                if len(y0.shape) == 1:
                    y0 = jnp.expand_dims(jnp.expand_dims(y0, 0), -1)
                elif len(y0.shape) == 2:
                    y0 = jnp.expand_dims(y0, -1)
                y0 = jnp.einsum('ijk,lmj->ilm', y0, kernels)
                y0 = self.act_fn(y0)
                if y0.shape[0] == 1:
                    y0 = jnp.squeeze(y0, 0)
                if len(y0.shape) == 2:
                    out = jnp.einsum('ij,ij->ij', data_adj, y0).sum(-1)
                elif len(y0.shape) == 3:
                    out = jnp.einsum('aij,aij->aij', data_adj, y0).sum(-1)
                out = jnp.squeeze(bias, -1) - out  # B*N
            else:
                if len(y0.shape) == 1:
                    y0 = jnp.expand_dims(y0, 0)
                for kernel in kernels:
                    y0 = jnp.einsum('ij,jk->ik', y0, kernel)
                    y0= self.act_fn(y0)
                if len(data_adj.shape) == 2:
                    y1 = jnp.einsum('ik,jk->k', y0, data_adj)
                    out = self.act_fn(y1) + jnp.squeeze(bias, -1)
                elif len(data_adj.shape) == 3:
                    y1 = jnp.einsum('ik,ijk->ik', y0, data_adj)
                    out = self.act_fn(y1) + jnp.squeeze(bias, -1)
            return out, bias, data_adj

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(fn),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.01,  # ts[1] - ts[0],
            y0=y,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6, dtmax=0.1),
            saveat=diffrax.SaveAt(ts=ts),
            made_jump=True,

        )

        return solution.ys[
            0]  # jax.lax.reshape(solution.ys[0], (solution.ys[0].shape[1], solution.ys[0].shape[0], solution.ys[0].shape[-1]))
    ##############################

class Sigmoid(nn.Module):

    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))

##############################

class Tanh(nn.Module):

    def __call__(self, x):
        x_exp, neg_x_exp = jnp.exp(x), jnp.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)

##############################

class ReLU(nn.Module):

    def __call__(self, x):
        return jnp.maximum(x, 0)

##############################

class LeakyReLU(nn.Module):
    alpha: float = 0.1

    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)

##############################

class ELU(nn.Module):

    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)

##############################

class Swish(nn.Module):

    def __call__(self, x):
        return x * nn.sigmoid(x)

##############################

class Sine(nn.Module):

    def __call__(self, x):
        return jnp.sin(x)

##############################

class Identity(nn.Module):

    def __call__(self, x):
        return x

act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
    "sine": Sine,
    "identity": Identity
    }