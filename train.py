import os

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
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


act_fn = act_fn_by_name['sine']()

os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_num}"

path = os.getcwd()
CHECKPOINT_PATH = f"{path}/saved_models/kuramoto_jax_{normal}normal"\
                  + f"{num_epochs}epochs_" \
                  + f"{act_fn}_actfn_" + f"{hdims}hdims"
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
    inp = (ys[:,0],bias, data_adj)
    return train_loader, eval_loader ,model_key, ts, inp

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
              path):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(model_name, model_class, model_hparams, optimizer_name, optimizer_hparams, ts, inp,
                            model_key, path)
    trainer.train_model(train_loader = train_loader, eval_loader = eval_loader, ts=ts, num_epochs=num_epochs, ratios = ratios)
    if trainer.checkpoint_exists():  # continue training if pretrained model exists
        trainer.load_model()

    else:
        trainer.load_model(pretrained=True)
    # Test trained model
    val_loss = trainer.eval_model(ts, eval_loader)
    return trainer, {'val': val_loss}

def train(path = CHECKPOINT_PATH):
    train_loader, eval_loader ,model_key, ts, inp = loader()
    print(f'path is {CHECKPOINT_PATH}')
    if model_class == 'NeuralODE':
        model = NeuralODE
    trainer, val_loss = train_ode(model_name="nODE",
                                  model_class=model,  # NeuralODE
                                  model_hparams={"node_size": node_size, "act_fn": act_fn, "normal" : normal, "hdims": hdims},
                                  optimizer_name=optimizer_name,
                                  optimizer_hparams={"lr": lr, "weight_decay": weight_decay},
                                  inp= inp,
                                  ts=ts,
                                  model_key=model_key,
                                  train_loader = train_loader,
                                  eval_loader = eval_loader,
                                  num_epochs=num_epochs,
                                  path = path)

    return trainer, val_loss


if __name__ == '__main__':
    train()