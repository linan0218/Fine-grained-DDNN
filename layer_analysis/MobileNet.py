from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
import time
import matplotlib.image as img_read
import datetime
"""Single-GPU AlexNet without partition toward the channel axis."""

class MobileNet(chainer.Chain):

    def __init__(self):
        super(MobileNet, self).__init__()
        with self.init_scope():
            self.convBN1 = L.Convolution2D(3, 32, 3, stride=1, pad=1, nobias=True)
            
            self.convDW1_1 = L.Convolution2D(32, 32, 3, stride=1, pad=1, nobias=True)
            self.convDW1_2 = L.Convolution2D(32, 64, 3, stride=1, pad=1, nobias=True)

            self.convDW2_1 = L.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True)
            self.convDW2_2 = L.Convolution2D(64, 128, 3, stride=1, pad=1, nobias=True)

            self.convDW3_1 = L.Convolution2D(128, 128, 3, stride=1, pad=1, nobias=True)
            self.convDW3_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1, nobias=True)
            self.convDW4_1 = L.Convolution2D(128, 128, 3, stride=2, pad=1, nobias=True)
            self.convDW4_2 = L.Convolution2D(128, 256, 3, stride=1, pad=1, nobias=True)

            self.convDW5_1 = L.Convolution2D(256, 256, 3, stride=1, pad=1, nobias=True)
            self.convDW5_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1, nobias=True)
            self.convDW6_1 = L.Convolution2D(256, 256, 3, stride=2, pad=1, nobias=True)
            self.convDW6_2 = L.Convolution2D(256, 512, 3, stride=1, pad=1, nobias=True)

            self.convDW7891011_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1, nobias=True)
            self.convDW7891011_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, nobias=True)

            self.convDW12_1 = L.Convolution2D(512, 512, 3, stride=2, pad=1, nobias=True)
            self.convDW12_2 = L.Convolution2D(512, 1024, 3, stride=1, pad=1, nobias=True)

            self.convDW13_1 = L.Convolution2D(1024, 1024, 3, stride=2, pad=1, nobias=True)
            self.convDW13_2 = L.Convolution2D(1024, 1024, 3, stride=1, pad=1, nobias=True)

            self.fc = L.Linear(None, 10)

    def __call__(self, x):

        h = self.convBN1(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        
        h = self.convDW1_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW1_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)

        h = self.convDW2_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW2_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)

        h = self.convDW3_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW3_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW4_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW4_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)

        h = self.convDW5_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW5_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW6_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW6_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)

        h = self.convDW7891011_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW7891011_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)

        h = self.convDW12_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW12_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW13_1(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = self.convDW13_2(h)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        begin = datetime.datetime.now()
        h = self.fc(h)
        end  = datetime.datetime.now()
        print((end - begin).total_seconds() * 1000, "ms")

        return h

if __name__ == '__main__':
    mobileNet_cifar10 = MobileNet()
    serializers.save_npz('part.model', mobileNet_cifar10)
    
    from _tool import chainerDataset
    _, _, x_test, y_test = chainerDataset.get_chainer_cifar10()
    # print(x_test[0].shape)
    output = mobileNet_cifar10(x_test[0].reshape(1,3,32, 32))
    import cPickle as pickle
    pickle.dump(output, open('output.txt','wb'))
