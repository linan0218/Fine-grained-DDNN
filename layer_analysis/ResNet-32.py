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

class ResNet32(chainer.Chain):
    def __init__(self):
        super(ResNet32, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, 3, stride=1, pad=0)


            self.conv2_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1)

            self.conv3_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1)

            self.conv4_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1)

            self.conv5_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1)
            self.conv5_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1)

            self.conv6_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1)
            self.conv6_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1)


            self.conv7_1 = L.Convolution2D(None, 32, 3, stride=1, pad=1)
            self.conv7_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1)

            self.conv8_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1)
            self.conv8_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1)

            self.conv9_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1)
            self.conv9_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1)

            self.conv10_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1)
            self.conv10_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1)

            self.conv11_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1)
            self.conv11_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1)


            self.conv12_1 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv12_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv13_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1)
            self.conv13_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv14_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1)
            self.conv14_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv15_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1)
            self.conv15_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv16_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1)
            self.conv16_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)


            self.fc1=L.Linear(None, 10)


    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)


        h = self.conv2_1(h)
        h = F.local_response_normalization(h)
        h = self.conv2_2(h)
        h = F.local_response_normalization(h)

        h = self.conv3_1(h)
        h = F.local_response_normalization(h)
        h = self.conv3_2(h)
        h = F.local_response_normalization(h)

        h = self.conv4_1(h)
        h = F.local_response_normalization(h)
        h = self.conv4_2(h)
        h = F.local_response_normalization(h)

        h = self.conv5_1(h)
        h = F.local_response_normalization(h)
        h = self.conv5_2(h)
        h = F.local_response_normalization(h)

        h = self.conv6_1(h)
        h = F.local_response_normalization(h)
        h = self.conv6_2(h)
        h = F.local_response_normalization(h)


        h = self.conv7_1(h)
        h = F.local_response_normalization(h)
        h = self.conv7_2(h)
        h = F.local_response_normalization(h)

        h = self.conv8_1(h)
        h = F.local_response_normalization(h)
        h = self.conv8_2(h)
        h = F.local_response_normalization(h)

        h = self.conv9_1(h)
        h = F.local_response_normalization(h)
        h = self.conv9_2(h)
        h = F.local_response_normalization(h)

        h = self.conv10_1(h)
        h = F.local_response_normalization(h)
        h = self.conv10_2(h)
        h = F.local_response_normalization(h)

        h = self.conv11_1(h)
        h = F.local_response_normalization(h)
        h = self.conv11_2(h)
        h = F.local_response_normalization(h)


        h = self.conv12_1(h)
        h = F.local_response_normalization(h)
        h = self.conv12_2(h)
        h = F.local_response_normalization(h)

        h = self.conv13_1(h)
        h = F.local_response_normalization(h)
        h = self.conv13_2(h)
        h = F.local_response_normalization(h)

        h = self.conv14_1(h)
        h = F.local_response_normalization(h)
        h = self.conv14_2(h)
        h = F.local_response_normalization(h)

        h = self.conv15_1(h)
        h = F.local_response_normalization(h)
        h = self.conv15_2(h)
        h = F.local_response_normalization(h)

        h = self.conv16_1(h)
        h = F.local_response_normalization(h)
        h = self.conv16_2(h)
        h = F.local_response_normalization(h)


        h = F.average_pooling_2d(h, 2, stride=2)
        h = self.fc1(h)
        return h

# loading existing model
model = ResNet32()
serializers.load_npz('res32_cifar10.model', model)


# define pre_ResNet-32
class ResNet32_cifar(chainer.Chain):
    def __init__(self):
        super(ResNet32_cifar, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, 3, stride=1, pad=0, initialW=model.conv1_1.W.data, initial_bias=model.conv1_1.b.data)


            self.conv2_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv2_1.W.data, initial_bias=model.conv2_1.b.data)
            self.conv2_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv2_2.W.data, initial_bias=model.conv2_2.b.data)

            self.conv3_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv3_1.W.data, initial_bias=model.conv3_1.b.data)
            self.conv3_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv3_2.W.data, initial_bias=model.conv3_2.b.data)

            self.conv4_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv4_1.W.data, initial_bias=model.conv4_1.b.data)
            self.conv4_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv4_2.W.data, initial_bias=model.conv4_2.b.data)

            self.conv5_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv5_1.W.data, initial_bias=model.conv5_1.b.data)
            self.conv5_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv5_2.W.data, initial_bias=model.conv5_2.b.data)

            self.conv6_1 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv6_1.W.data, initial_bias=model.conv6_1.b.data)
            self.conv6_2 = L.Convolution2D(None, 16, 3, stride=1, pad=1, initialW=model.conv6_2.W.data, initial_bias=model.conv6_2.b.data)


            self.conv7_1 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv7_1.W.data, initial_bias=model.conv7_1.b.data)
            self.conv7_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv7_2.W.data, initial_bias=model.conv7_2.b.data)

            self.conv8_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1, initialW=model.conv8_1.W.data, initial_bias=model.conv8_1.b.data)
            self.conv8_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv8_2.W.data, initial_bias=model.conv8_2.b.data)

            self.conv9_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1, initialW=model.conv9_1.W.data, initial_bias=model.conv9_1.b.data)
            self.conv9_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv9_2.W.data, initial_bias=model.conv9_2.b.data)

            self.conv10_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1, initialW=model.conv10_1.W.data, initial_bias=model.conv10_1.b.data)
            self.conv10_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv10_2.W.data, initial_bias=model.conv10_2.b.data)

            self.conv11_1 = L.Convolution2D(None, 32, 3, stride=2, pad=1, initialW=model.conv11_1.W.data, initial_bias=model.conv11_1.b.data)
            self.conv11_2 = L.Convolution2D(None, 32, 3, stride=1, pad=1, initialW=model.conv11_2.W.data, initial_bias=model.conv11_2.b.data)


            self.conv12_1 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv12_1.W.data, initial_bias=model.conv12_1.b.data)
            self.conv12_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv12_2.W.data, initial_bias=model.conv12_2.b.data)

            self.conv13_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1, initialW=model.conv13_1.W.data, initial_bias=model.conv13_1.b.data)
            self.conv13_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv13_2.W.data, initial_bias=model.conv13_2.b.data)

            self.conv14_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1, initialW=model.conv14_1.W.data, initial_bias=model.conv14_1.b.data)
            self.conv14_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv14_2.W.data, initial_bias=model.conv14_2.b.data)

            self.conv15_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1, initialW=model.conv15_1.W.data, initial_bias=model.conv15_1.b.data)
            self.conv15_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv15_2.W.data, initial_bias=model.conv15_2.b.data)

            self.conv16_1 = L.Convolution2D(None, 64, 3, stride=2, pad=1, initialW=model.conv16_1.W.data, initial_bias=model.conv16_1.b.data)
            self.conv16_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=model.conv16_2.W.data, initial_bias=model.conv16_2.b.data)


            self.fc1=L.Linear(None, 10)


    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)


        h = self.conv2_1(h)
        h = F.local_response_normalization(h)
        h = self.conv2_2(h)
        h = F.local_response_normalization(h)

        h = self.conv3_1(h)
        h = F.local_response_normalization(h)
        h = self.conv3_2(h)
        h = F.local_response_normalization(h)

        h = self.conv4_1(h)
        h = F.local_response_normalization(h)
        h = self.conv4_2(h)
        h = F.local_response_normalization(h)

        h = self.conv5_1(h)
        h = F.local_response_normalization(h)
        h = self.conv5_2(h)
        h = F.local_response_normalization(h)

        h = self.conv6_1(h)
        h = F.local_response_normalization(h)
        h = self.conv6_2(h)
        h = F.local_response_normalization(h)


        h = self.conv7_1(h)
        h = F.local_response_normalization(h)
        h = self.conv7_2(h)
        h = F.local_response_normalization(h)

        h = self.conv8_1(h)
        h = F.local_response_normalization(h)
        h = self.conv8_2(h)
        h = F.local_response_normalization(h)

        h = self.conv9_1(h)
        h = F.local_response_normalization(h)
        h = self.conv9_2(h)
        h = F.local_response_normalization(h)

        h = self.conv10_1(h)
        h = F.local_response_normalization(h)
        h = self.conv10_2(h)
        h = F.local_response_normalization(h)

        h = self.conv11_1(h)
        h = F.local_response_normalization(h)
        h = self.conv11_2(h)
        h = F.local_response_normalization(h)


        h = self.conv12_1(h)
        h = F.local_response_normalization(h)
        h = self.conv12_2(h)
        h = F.local_response_normalization(h)

        h = self.conv13_1(h)
        h = F.local_response_normalization(h)
        h = self.conv13_2(h)
        h = F.local_response_normalization(h)

        h = self.conv14_1(h)
        h = F.local_response_normalization(h)
        h = self.conv14_2(h)
        h = F.local_response_normalization(h)


        h = self.conv15_1(h)
        h = F.local_response_normalization(h)
        h = self.conv15_2(h)
        h = F.local_response_normalization(h)

        h = self.conv16_1(h)
        h = F.local_response_normalization(h)
        h = self.conv16_2(h)
        h = F.local_response_normalization(h)


        begin = datetime.datetime.now()
        h = self.fc1(h)
        end  = datetime.datetime.now()
        print((end - begin).total_seconds() * 1000, "ms")
        
        return h


if __name__ == '__main__':
    ResNet32_cifar10 = ResNet32_cifar()
    serializers.save_npz('part.model', ResNet32_cifar10)    
    
    from _tool import chainerDataset
    _, _, x_test, y_test = chainerDataset.get_chainer_cifar10()
    # print(x_test[0].shape)
    output = ResNet32_cifar10(x_test[0].reshape(1,3,32,32))
    import cPickle as pickle
    pickle.dump(output, open('output.txt','wb'))
