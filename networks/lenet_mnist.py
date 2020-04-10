from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F

def get_network(percentTrainKeeps=1, n_class=10):
    network = [
        L.Convolution2D(1, 5, 5, stride=1, pad=3),
        Branch([L.Linear(None, n_class)]),  # 1
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 2
        L.Convolution2D(5, 10, 5, stride=1, pad=3),
        Branch([L.Linear(None, n_class)]),  # 3
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 4
        L.Convolution2D(10, 20, 5, stride=1, pad=3),
        Branch([L.Linear(None, n_class)]),  # 5
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 6
        L.Linear(None, 84),  # 720, 84
        Branch([L.Linear(None, n_class)])  # 84, 10
    ]
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
