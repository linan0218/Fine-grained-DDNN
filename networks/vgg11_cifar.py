from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F

# VGG-11

def get_network(percentTrainKeeps=1, n_class=10):
    network = [
        L.Convolution2D(None, 64, 3, stride=1, pad=1),
        L.BatchNormalization(64),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 1
        L.Convolution2D(64, 64, 3, stride=1, pad=1),
        L.BatchNormalization(64),
        FL(F.ReLU()),
        FL(F.max_pooling_2d, 2, 2),
        SL(FL(F.dropout, 0.25, train=True)),
        Branch([L.Linear(None, n_class)]),  # 2

        L.Convolution2D(64, 128, 3, stride=1, pad=1),
        L.BatchNormalization(128),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 3
        L.Convolution2D(128, 128, 3, stride=1, pad=1),
        L.BatchNormalization(128),
        FL(F.ReLU()),
        FL(F.max_pooling_2d, 2, 2),
        SL(FL(F.dropout, 0.25, train=True)),
        Branch([L.Linear(None, n_class)]),  # 4

        L.Convolution2D(128, 256, 3, stride=1, pad=1),
        L.BatchNormalization(256),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 5
        L.Convolution2D(256, 256, 3, stride=1, pad=1),
        L.BatchNormalization(256),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 6
        L.Convolution2D(256, 256, 3, stride=1, pad=1),
        L.BatchNormalization(256),
        FL(F.ReLU()),
        Branch([L.Linear(None, n_class)]),  # 7
        L.Convolution2D(256, 256, 3, stride=1, pad=1),
        L.BatchNormalization(256),
        FL(F.ReLU()),
        FL(F.max_pooling_2d, 2, 2),
        SL(FL(F.dropout, 0.25, train=True)),
        Branch([L.Linear(None, n_class)]),  # 8

        L.Linear(None, 4096),
        FL(F.ReLU()),
        SL(FL(F.dropout, 0.5, train=True)),
        Branch([L.Linear(None, n_class)]),  # 9

        L.Linear(4096, 4096),
        FL(F.ReLU()),
        SL(FL(F.dropout, 0.5, train=True)),

        #L.Linear(4096, 1000),
        Branch([L.Linear(None, n_class)])  # 10
    ]
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
