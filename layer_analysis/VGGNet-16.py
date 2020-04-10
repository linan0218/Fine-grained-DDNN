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

class VGG16Net(chainer.Chain):
    def __init__(self):
        super(VGG16Net, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1)

            self.conv8=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv9=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv11=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv13=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc14=L.Linear(None, 4096)
            self.fc15=L.Linear(None, 4096)
            self.fc16=L.Linear(None, 10)


    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv7(h))), 2, stride=2)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv10(h))), 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv13(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc14(h)))
        h = F.dropout(F.relu(self.fc15(h)))
        h = self.fc16(h)
        return h


# loading existing model
model = VGG16Net()
serializers.load_npz('vgg16_cifar10.model', model)


# define pre_VGG16
class VGG16Net_cifar(chainer.Chain):
    def __init__(self):
        super(VGG16Net_cifar, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv1.W.data, initial_bias=model.conv1.b.data)
            self.conv2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv2.W.data, initial_bias=model.conv2.b.data)

            self.conv3 = L.Convolution2D(None, 128, 3, stride=1, pad=1, initialW=model.conv3.W.data, initial_bias=model.conv3.b.data)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=1, pad=1, initialW=model.conv4.W.data, initial_bias=model.conv4.b.data)

            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1, initialW=model.conv5.W.data, initial_bias=model.conv5.b.data)
            self.conv6 = L.Convolution2D(None, 256, 3, stride=1, pad=1, initialW=model.conv6.W.data, initial_bias=model.conv6.b.data)
            self.conv7 = L.Convolution2D(None, 256, 3, stride=1, pad=1, initialW=model.conv7.W.data, initial_bias=model.conv7.b.data)

            self.conv8 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv8.W.data, initial_bias=model.conv8.b.data)
            self.conv9 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv9.W.data, initial_bias=model.conv9.b.data)
            self.conv10 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv10.W.data, initial_bias=model.conv10.b.data)

            self.conv11 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv11.W.data, initial_bias=model.conv11.b.data)
            self.conv12 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv12.W.data, initial_bias=model.conv12.b.data)
            self.conv13 = L.Convolution2D(None, 512, 3, stride=1, pad=1, initialW=model.conv13.W.data, initial_bias=model.conv13.b.data)

            self.fc14=L.Linear(None, 4096)
            self.fc15=L.Linear(None, 4096)
            self.fc16=L.Linear(None, 10)


    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)

        h = self.conv2(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.conv5(h)
        h = F.relu(h)
        h = self.conv6(h)
        h = F.relu(h)
        h = self.conv7(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.conv8(h)
        h = F.relu(h)
        h = self.conv9(h)
        h = F.relu(h)
        h = self.conv10(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.conv11(h)
        h = F.relu(h)
        h = self.conv12(h)
        h = F.relu(h)
        h = self.conv13(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.fc14(h)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.fc15(h)
        h = F.relu(h)
        h = F.dropout(h)
        begin = datetime.datetime.now()
        h = self.fc16(h)
        end  = datetime.datetime.now()
        print((end - begin).total_seconds() * 1000, "ms")
        return h



if __name__ == '__main__':
    VGG16Net_cifar10 = VGG16Net_cifar()
    serializers.save_npz('part.model', VGG16Net_cifar10)    
    
    from _tool import chainerDataset
    _, _, x_test, y_test = chainerDataset.get_chainer_cifar10()
    # print(x_test[0].shape)
    output = VGG16Net_cifar10(x_test[0].reshape(1,3,32,32))
    import cPickle as pickle
    pickle.dump(output, open('output.txt','wb'))
