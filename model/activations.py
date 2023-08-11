import jax
import jax.numpy as jnp
import equinox as eqx
jax.config.update("jax_enable_x64", True)




class Sigmoid(eqx.Module):

    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))

##############################

class Tanh(eqx.Module):

    def __call__(self, x):
        x_exp, neg_x_exp = jnp.exp(x), jnp.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)

##############################

class ReLU(eqx.Module):

    def __call__(self, x):
        return jnp.maximum(x, 0)

##############################

class LeakyReLU(eqx.Module):
    alpha: float = 0.1

    def __call__(self, x):
        return jnp.where(x > 0, x, self.alpha * x)

##############################

class ELU(eqx.Module):

    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)

##############################

class Swish(eqx.Module):

    def __call__(self, x):
        return x * jax.nn.sigmoid(x)

##############################

class Sine(eqx.Module):

    def __call__(self, x):
        return jnp.sin(x)

##############################

class Identity(eqx.Module):

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
