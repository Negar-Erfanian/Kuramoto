import jax.numpy as jnp
import jax.random as jrandom


def dataloader(arrays, batch_size, *, key):
  dataset_size = arrays[0].shape[0]
  assert all(array.shape[0] == dataset_size for array in arrays)
  indices = jnp.arange(dataset_size)
  permutation = jrandom.permutation(key, indices)
  train_batches = [[array[permutation[i:i+batch_size]] for array in arrays] for i in range(0, dataset_size, batch_size)]
  return train_batches