import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import argparse


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def main(args):
    device = torch.device("cuda")
    batch_size, lr, num_epochs = args.batch_size, args.lr, args.epochs

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    net.load_state_dict(torch.load("mlp.params"))

    loss = nn.CrossEntropyLoss(reduction='none')

    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print(net.state_dict())
    print(net)
    torch.save(net.state_dict(), 'mlp.params')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=int, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    cur_arg = parser.parse_args()

    main(cur_arg)
