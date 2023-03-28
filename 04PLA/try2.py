import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

lr = 0.03
bs = 30
epoches = 10

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=bs)

d2l.train_ch3(net, train_iter, test_iter, loss, epoches, trainer)
d2l.plt.show()
