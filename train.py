import matplotlib.pyplot as plt

from utils import *
from model.neuralODE import *
from model.trainer import *
from data.Kuramotodata import *
from data.load_data import *

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

args, _ = get_args()
model_name = args.model_name
model_class = args.model_class
node_size = args.node_size #50
hdims = args.hdims
optimizer_name = args.optimizer_name
num_epochs = args.num_epochs
gpu_num = args.gpu_num
seed = args.seed
data_size = args.data_size
train_size = args.train_size
batch_size = args.batch_size
normal = args.normal
lr = args.lr
weight_decay = args.weight_decay
train = args.train
losstype = args.losstype

ratios = list(np.linspace(0.01,1,100))


act_fn = act_fn_by_name['sine']()
actfn = act_fn_by_name['relu']()

os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_num}"

path = os.getcwd()
CHECKPOINT_PATH = f"{path}/saved_models/kuramoto_jax_{normal}normal"\
                  + f"{num_epochs}epochs_" +\
                  f"{batch_size}batchsize_" +\
                  f"{node_size}nodesize_" \
                  + f"{act_fn}actfn_" + f"{hdims}hdims_" + f"{losstype}loss"
#CHECKPOINT_PATH = f"{path}/saved_models/test"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def loader():
    key = jrandom.PRNGKey(seed)
    model_key, loader_key = jrandom.split(key, 2)

    ts, ys, theta, bias, data_adj, mat = KuramotoDataset(dataset_size=data_size, node_num=node_size,
                                                         seed=seed).get_data()
    train_data = (ys[:train_size], bias[:train_size], data_adj[:train_size])
    eval_data = (ys[train_size:], bias[train_size:], data_adj[train_size:])
    train_loader = dataloader(train_data, batch_size=batch_size, key=loader_key)
    eval_loader = dataloader(eval_data, batch_size=batch_size, key=loader_key)
    inp = (ys[:,0], (bias, data_adj))
    return train_loader, eval_loader ,model_key, ts, inp, mat

def train_ode(model_name,
              model_class,  # NeuralODE
              model_hparams,
              optimizer_name,
              optimizer_hparams,
              inp,  # (ys[:,0],bias, data_adj)
              ts,
              model_key,
              train_loader,
              eval_loader,
              num_epochs,
              path,
              train,
              losstype):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(model_name, model_class, model_hparams, optimizer_name, optimizer_hparams, ts, inp,
                            model_key, path, losstype)
    if train:
        trainer.train_model(train_loader = train_loader, eval_loader = eval_loader, ts=ts, num_epochs=num_epochs, ratios = ratios)
    else:  # continue training if pretrained model exists
        trainer.load_model()

    # Test trained model
    val_loss = trainer.eval_model(ts, eval_loader)
    return trainer, {'val': val_loss}

def trainmodel(path = CHECKPOINT_PATH):
    train_loader, eval_loader ,model_key, ts, inp, mat = loader()
    print(f'path is {CHECKPOINT_PATH}')
    if model_class == 'NeuralODE':
        model = NeuralODE
    trainer, val_loss = train_ode(model_name="nODE",
                                  model_class=model,  # NeuralODE
                                  model_hparams={"node_size": node_size, "act_fn": act_fn, "normal" : normal, "hdims": hdims, "actfn": actfn},
                                  optimizer_name=optimizer_name,
                                  optimizer_hparams={"lr": lr, "weight_decay": weight_decay},
                                  inp= inp,
                                  ts=ts,
                                  model_key=model_key,
                                  train_loader = train_loader,
                                  eval_loader = eval_loader,
                                  num_epochs=num_epochs,
                                  path = path,
                                  train = train,
                                  losstype = losstype)
    if not train:
        idx = 0
        parameters = jax.tree_util.tree_leaves(trainer.state.params)
        print('The learned kernels have the following shapes:', ', '.join([str(p.shape) for p in parameters]))
        print('Overall number of parameters:', sum([np.prod(p.shape) for p in parameters]))
        print(jax.tree_map(lambda p: p.shape, trainer.state.params))
        visualization(train_loader, trainer, ts, idx, loader_name = "train")
        visualization(eval_loader, trainer, ts, idx, loader_name = "eval")
        if hdims==None:
            weight_viz(trainer, mat)
    return trainer, val_loss


def visualization(loader, trainer, ts, idx, loader_name):
    ys, b, adj = next(iter(loader))
    params = trainer.state.params
    batch = (ys[:, 0], (b, adj))
    y_pred = jax.vmap(trainer.state.apply_fn, in_axes=(None, None, 0))(params, ts, batch)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(jnp.sin(ys[idx]))
    axes[0, 0].set_title('Real data Sine')
    axes[0, 1].plot(ys[idx])
    axes[0, 1].set_title('Real data phase')
    axes[1, 0].plot(jnp.sin(y_pred[idx]))
    axes[1, 0].set_title('Predicted data Sine')
    axes[1, 1].plot(y_pred[idx])
    axes[1, 1].set_title('Predicted data phase')
    plt.tight_layout()
    plt.savefig(f'{CHECKPOINT_PATH}/Kuramoto-{loader_name}.png')
    plt.show()

def weight_viz(trainer, mat):
    fig, axes = plt.subplots(1, 3, figsize=(10, 8))
    axes[0].imshow(trainer.state.params['params']['kernel'][0])
    axes[0].set_title('Learned weights')
    axes[1].imshow(mat[0][0])
    axes[1].set_title('True weights')
    axes[2].imshow(jnp.abs(mat[0][0] - trainer.state.params['params']['kernel'][0]))
    axes[2].set_title('difference')
    plt.tight_layout()
    plt.savefig(f'{CHECKPOINT_PATH}/Kuramoto_weights.png')
    plt.show()


if __name__ == '__main__':
    trainer, val_loss = trainmodel()