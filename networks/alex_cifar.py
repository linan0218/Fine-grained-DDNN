from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.functions as F
import chainer.links as L


def norm():
    return [FL(F.relu), FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75)]


conv = lambda n: [L.Convolution2D(n, 32, 3, pad=1, stride=1), FL(F.relu)]
cap = lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]


def gen_2b(n_class):
    network = [
        L.Convolution2D(3, 32, 5, pad=2, stride=1),
        FL(F.relu),
        Branch([L.Linear(None, n_class)]),  # 1
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
        Branch([L.Linear(None, n_class)]),  # 2
        L.Convolution2D(32, 64, 5, pad=2, stride=1),
        FL(F.relu),
        Branch([L.Linear(None, n_class)]),  # 3
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization, n=3, alpha=5e-05, beta=0.75),
        Branch([L.Linear(None, n_class)]),  # 4
        L.Convolution2D(64, 96, 3, pad=1, stride=1),
        FL(F.relu),
        Branch([L.Linear(None, n_class)]),  # 5
        L.Convolution2D(96, 96, 3, pad=1, stride=1),
        FL(F.relu),
        Branch([L.Linear(None, n_class)]),  # 6
        L.Convolution2D(96, 64, 3, pad=1, stride=1),
        FL(F.relu),
        Branch([L.Linear(None, n_class)]),  # 7
        FL(F.max_pooling_2d, 3, 2),
        Branch([L.Linear(None, n_class)]),  # 8
        L.Linear(1024, 256),
        FL(F.relu),
        SL(FL(F.dropout, 0.5, train=True)),
        Branch([L.Linear(None, n_class)]),  # 9
        L.Linear(256, 128),
        FL(F.relu),
        SL(FL(F.dropout, 0.5, train=True)),
        Branch([L.Linear(128, n_class)])
    ]
    
    print((np.array(network)).shape)
    return network


def get_network(n_class=10, percentTrainKeeps=1):
    network = gen_2b(n_class)
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
