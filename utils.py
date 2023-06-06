import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument("--hdims", type=str, default="32", help='layer size for normal NN')
    parser.add_argument('--model_name', type=str, default="nODE", help='model_name')
    parser.add_argument('--model_class', type=str, default="NeuralODE", help='model_class')
    parser.add_argument('--node_size', type=int, default=50, help='node_size')
    parser.add_argument('--optimizer_name', type=str, default="sgd", help='optimizer') #'adam', 'adamw'
    parser.add_argument('--gpu-num', type=int, default=2, help='gpu-num')
    parser.add_argument('--seed', type=int, default=758493, help='seed')
    parser.add_argument('--normal', type=str, default=True, help='normal NN')
    parser.add_argument('--data_size', type=int, default=256, help='data_size')
    parser.add_argument('--train_size', type=int, default=160, help='train_size')
    parser.add_argument('--lr', type=int, default=1e-1, help='learning_rate')
    parser.add_argument('--weight-decay', type=int, default=1e-4, help='weight_decay')

    args, unparsed = parser.parse_known_args()
    return args, unparsed
