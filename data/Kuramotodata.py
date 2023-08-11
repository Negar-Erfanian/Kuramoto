import diffrax


from diffrax import *
import jax

import jax.numpy as jnp
import jax.random as jrandom

import numpy as np
import networkx as nx
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
sns.reset_orig()

class KuramotoDataset():

    def __init__(self, dataset_size, node_num, seed, datatype, weight_params=(0, 0.1), bias_params=(1, 0.1), prob=0.1, t0=0, t1=4,
                 tt=500):
        """
        Inputs:
            size - Number of data points we want to generate
            seed - The seed to use to create the PRNG state with which we want to generate the data points
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        self.dataset_size = dataset_size
        self.weight_params = weight_params
        self.bias_params = bias_params
        self.prob = prob
        self.node_num = node_num
        self.seed = seed
        self.key = jrandom.PRNGKey(seed)
        self.datatype = datatype
        self.t = (t0, t1, tt)

    def weighted_graph(self, key):
        #print(f'shape is {jrandom.uniform(key, shape = (self.node_num, self.node_num))}')
        arr = jrandom.uniform(key, shape = (self.node_num, self.node_num))
        weights = jrandom.normal(key, shape = (self.node_num, self.node_num))*self.weight_params[1] + self.weight_params[0]
        arr = jnp.matmul(jnp.array(arr>=self.prob, int), weights)
        '''print(f'{arr.shape}')
        #np.random.seed(self.seed)
        G = nx.Graph()  # Create a graph object called G
        node_list = [i for i in range(self.node_num)]
        edge_list = []
        for node in node_list:
            G.add_node(node)

        # randomly conecting nodes (constructing edges)
        for i in range(self.node_num):
            for j in range(i + 1, self.node_num):
                p = np.random.uniform(0, 1)
                if p >= self.prob:
                    edge_list.append((i, j))

        edge_num = len(edge_list)

        weights = np.abs(np.random.normal(self.weight_params[0], self.weight_params[1],
                                          edge_num))  # weights associated with the connected edges
        for w, edge in enumerate(edge_list):
            G.add_edge(edge[0], edge[1], weight=weights[w])

        data_adj = jnp.array(nx.adjacency_matrix(G).todense())'''
        return arr

    def _get_data(self, ts, key):
        theta = jrandom.normal(key, shape=(self.node_num, 1)) * 2 * jnp.pi  # initial phase \theta between 0 and \pi
        bias = jrandom.normal(key, shape=(self.node_num, 1)) * self.bias_params[1] + self.bias_params[
            0]  # initial inherent frequency \w chosen from a gaussian distribution
        data_adj = self.weighted_graph(key)
        mat = jnp.zeros((self.node_num, self.node_num,
                         self.node_num))  # wight matrix that couplutes the phase difference when multiplied by \theta
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(mat.shape[2]):
                    if i != j:
                        if k == i:
                            mat = mat.at[i, j, k].set(1)
                        if k == j:
                            mat = mat.at[i, j, k].set(-1)

        def f(t, y, args):

            p = jnp.einsum('ijk,lmj->ilm', jnp.expand_dims(y, 0), mat)
            B = jnp.sin(p)  # B*n*n

            out = jnp.squeeze(jnp.expand_dims(jnp.einsum('aij,aij->aij', jnp.expand_dims(data_adj, 0), B).sum(-1), -1),
                              0)
            out = bias - out

            return out

        if self.datatype == "deterministic":
            solver = diffrax.Dopri5()
            dt0 = 0.01
            saveat = diffrax.SaveAt(ts=ts)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, theta, saveat=saveat,
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-6, dtmax=0.1))
            ys = sol.ys
        elif self.datatype == "stochastic":
            diffusion = lambda t, y, args: 0.05 * t
            brownian_motion = VirtualBrownianTree(ts[0], ts[-1], tol=1e-5, shape=(), key=jrandom.PRNGKey(0))
            terms = MultiTerm(ODETerm(f), ControlTerm(diffusion, brownian_motion))
            solver = diffrax.Dopri5()
            dt0 = 0.01
            saveat = diffrax.SaveAt(ts=ts)
            sol = diffrax.diffeqsolve(terms, solver, ts[0], ts[-1], dt0, theta,
                                      saveat=saveat,
                                      stepsize_controller=PIDController(rtol=1e-5,
                                                                        atol=1e-5,
                                                                        dtmax=0.1,
                                                                        error_order=0.5))
            ys = sol.ys
        return ys, theta, bias, data_adj, mat

    def get_data(self):
        ts = jnp.linspace(self.t[0], self.t[1], self.t[2])
        key = jrandom.split(self.key, self.dataset_size)
        ys, theta, bias, data_adj, mat = jax.vmap(lambda key: self._get_data(ts, key=key))(key)
        return ts, jnp.squeeze(ys, -1), theta, bias, data_adj, mat

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.dataset_size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        _, data_point, theta, bias, data_adj, _ = self.get_data()
        return data_point[idx], theta[idx], bias[idx], data_adj[idx]

if __name__ == '__main__':
    dataclass = KuramotoDataset(dataset_size = 32, node_num = 100, seed = 100, type = "deterministic", weight_params=(0, 0.1), bias_params=(1, 0.1), prob=0.1, t0=0, t1=4, tt=500)
    ts, ys, theta, bias, data_adj, mat = dataclass.get_data()