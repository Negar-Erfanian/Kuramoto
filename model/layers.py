import jax
import jax.numpy as jnp
from .activations import *
import equinox as eqx
jax.config.update("jax_enable_x64", True)





def gcn_layer(adj_matrix, node_features, kernels, activation=LeakyReLU()):
    # Perform propagation using symmetrically normalized adjacency matrix
    norm_adj_matrix = adj_matrix + jnp.eye(adj_matrix.shape[0])  # Add self-loops
    degree_inv_sqrt = 1.0 / jnp.sqrt(jnp.sum(norm_adj_matrix, axis=1))
    degree_inv_sqrt_matrix = jnp.diag(degree_inv_sqrt)
    normalized_adj_matrix = jnp.matmul(jnp.matmul(degree_inv_sqrt_matrix, norm_adj_matrix), degree_inv_sqrt_matrix)

    # Perform graph convolution
    for kernel in kernels:
        node_features = jnp.matmul(normalized_adj_matrix, node_features)
        node_features = jnp.matmul(node_features, kernel)

        # Apply activation function
        if activation is not None:
            node_features = activation(node_features)

    return node_features