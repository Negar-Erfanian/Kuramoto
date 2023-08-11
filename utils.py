import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument("--hdims", type=str, default="32-64-32", help='layer size for normal NN') #"32-64"
    parser.add_argument('--model_name', type=str, default="nODE", help='model_name')
    parser.add_argument('--model_class', type=str, default="NeuralODE", help='model_class')
    parser.add_argument('--node_size', type=int, default=50, help='node_size')
    parser.add_argument('--optimizer_name', type=str, default="adam", help='optimizer') #'adam', 'adamw' 'adablf'
    parser.add_argument('--losstype', type=str, default="criterion", help='type of loss') #"criterion", "sin"
    parser.add_argument('--datatype', type=str, default="deterministic", help='type of data') #"deterministic", "stochastic"
    parser.add_argument('--gpu-num', type=int, default=3, help='gpu-num')
    parser.add_argument('--seed', type=int, default=758493, help='seed')
    parser.add_argument('--normal', type=str, default=True, help='normal NN')
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--notrain', dest='train', action='store_false')
    parser.add_argument('--data_size', type=int, default=256, help='data_size')
    parser.add_argument('--train_size', type=int, default=160, help='train_size')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning_rate')
    parser.add_argument('--dropoutrate', type=float, default=0.1, help='dropout_rate')
    parser.add_argument('--weight-decay', type=int, default=1e-4, help='weight_decay')

    args, unparsed = parser.parse_known_args()
    return args, unparsed
