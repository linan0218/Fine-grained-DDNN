from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.functions as F
import chainer.links as L

def ConvBN(inp, oup, stride):
    return [
        L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True),
        L.BatchNormalization(oup), # for cifar
        FL(F.relu)
    ]

def ConvDW(inp, oup, stride):
    return [
        L.Convolution2D(inp, inp, 3, stride=stride, pad=1, nobias=True),
        L.BatchNormalization(inp),
        FL(F.relu),
        L.Convolution2D(inp, oup, 1, stride=1, pad=0, nobias=True),
        L.BatchNormalization(oup),
        FL(F.relu)
    ]


def gen_2b_cifar(n_class):
    network = ConvBN(3, 32, 1)

    network += ConvDW(32, 64, 1)

    network += ConvDW(64, 128, 1)
        
    network += ConvDW(128, 128, 1)
    network += ConvDW(128, 256, 2)
        
    network += ConvDW(256, 256, 1)
    network += ConvDW(256, 512, 2)

    network += ConvDW(512, 512, 1)
    network += ConvDW(512, 512, 1)
    network += ConvDW(512, 512, 1)
    network += ConvDW(512, 512, 1)
    network += ConvDW(512, 512, 1)

    network += ConvDW(512, 1024, 2)
    network += ConvDW(1024, 1024, 2)
        
    #network += [FL(F.average_pooling_2d, 7, 1)]
    network += [Branch([L.Linear(None, n_class)])]
    
    import numpy as np
    print("Final", (np.array(network)).shape)

    return network


def get_network(n_class=10, percentTrainKeeps=1):
    network = gen_2b_cifar(n_class)
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
