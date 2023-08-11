import diffrax
import jax
import jax.random as jrandom
jax.config.update("jax_enable_x64", True)
import os
path = os.getcwd()
from typing import Callable
from model.activations import *
from model.layers import *


## Imports for plotting
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
sns.reset_orig()

class NeuralODEgnn(eqx.Module):  # Here I will solve using the solver
    node_size: int
    hdims: str
    normal: bool
    act_fn: eqx.Module
    #batchnorm: eqx.nn.BatchNorm
    dropout: eqx.nn.Dropout
    kernels: list
    key2 : int


    def __init__(self, node_size, hdims, normal, dropoutrate, *, key, **kwargs):
        super(NeuralODEgnn).__init__(**kwargs)
        self.dropout= eqx.nn.Dropout(p=dropoutrate)
        self.node_size = node_size
        self.hdims = hdims
        self.normal = normal
        self.act_fn = Sine()
        key1, self.key2 = jrandom.split(key, 2)

        #self.batchnorm = eqx.nn.BatchNorm(self.node_size, axis_name=0)
        self.kernels_init(key1)




    def kernels_init(self, key):
        if self.hdims != None:
            hdims_list = [int(i) for i in self.hdims.split('-')]
        else:
            hdims_list = []
        kernels = []
        first_dim = self.node_size
        keys = jrandom.split(key, len(hdims_list)+1)
        if not self.normal:
            for i, dim in enumerate(hdims_list):
                kernels.append(jrandom.normal(keys[i], shape = (dim, self.node_size, first_dim)))
                first_dim = dim
            kernels.append(jrandom.normal(keys[-1], shape =(self.node_size, self.node_size, first_dim)))
        else:
            for i, dim in enumerate(hdims_list):
                kernels.append(jrandom.normal(keys[i], shape = (first_dim, dim)))
                first_dim = dim
            kernels.append(jrandom.normal(keys[-1], shape = (first_dim, 1)))
        self.kernels = kernels

    def __call__(self, ts, ys):
        y0, args = ys
        print(f'shapes are {y0.shape}, bias is {args[0].shape} and adj is {args[1].shape}')
        def fn(t, y, args):
            y0 = y
            bias, data_adj = args
            if not self.normal:
                if len(y0.shape) == 2:
                    y0 = jnp.expand_dims(y0, -1)
                elif len(y0.shape) == 1:
                    y0 = jnp.expand_dims(jnp.expand_dims(y0, -1), 0)
                for kernel in self.kernels:
                    y0 = jnp.einsum('ijk,lmj->ilm', y0, kernel)
                y0 = self.act_fn(y0)
                #self.batchnorm = eqx.nn.BatchNorm(y0, key=self.key2)
                y0 = self.dropout(y0, key = self.key2)
                if y0.shape[0] == 1:
                    y0 = jnp.squeeze(y0, 0)
                if len(y0.shape) == 2:
                    out = jnp.einsum('ij,ij->ij', data_adj, y0).sum(-1)
                elif len(y0.shape) == 3:
                    out = jnp.einsum('aij,aij->aij', data_adj, y0).sum(-1)
                out = jnp.squeeze(bias, -1) - out  # B*N
            else:
                out = gcn_layer(data_adj, jnp.expand_dims(y0, -1), self.kernels, activation=LeakyReLU())
                '''if len(y0.shape) == 1:
                    y0 = jnp.expand_dims(y0, 0)
                for kernel in self.kernels:
                    y0 = jnp.einsum('ij,jk->ik', y0, kernel)
                    y0= self.act_fn(y0)
                if len(data_adj.shape) == 2:
                    y1 = jnp.einsum('ik,jk->k', y0, data_adj)
                    out = self.act_fn(y1) + jnp.squeeze(bias, -1)
                elif len(data_adj.shape) == 3:
                    y1 = jnp.einsum('ik,ijk->ik', y0, data_adj)
                    out = self.act_fn(y1) + jnp.squeeze(bias, -1)'''
                out = jnp.squeeze(out, -1) + jnp.squeeze(bias, -1)
            print(f'out shape is{out.shape}')
            return out


        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(fn),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.01,  # ts[1] - ts[0],
            y0=y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6, dtmax=0.1),
            saveat=diffrax.SaveAt(ts=ts),
            made_jump=True,
        )
        # print(f'what is the shape of solution? {solution.ys[0].shape} and {solution.ys[1].shape} and {solution.ys[2].shape} and {len(solution.ys)}')
        return solution.ys

if __name__ == '__main__':
    batch_size = 30
    node_size = 50
    t_len = 500
    key1, key2, key3 = jrandom.split(jrandom.PRNGKey(1), 3)
    x = jrandom.normal(key1, shape =(batch_size,t_len,node_size))
    bias = jrandom.normal(key2, shape =(batch_size,node_size, 1))
    mat = jrandom.normal(key3, shape =(batch_size,node_size,node_size))
    args = (bias, mat)
    ts = jnp.linspace(0, 1, t_len)

    key = jrandom.PRNGKey(0)
    model_hparams = {"node_size": node_size,
                     "normal": False,
                     "hdims": None,
                     "dropoutrate": 0.1,
                     }
    model = NeuralODEgnn(**model_hparams, key = key)
    y_pred = jax.vmap(model, in_axes=(None, 0))(ts, (x[:, 0], args))
    print(f'model is {model}')
    print(f'y_pred is {y_pred.shape}')