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

class AlexNet(chainer.Chain):

    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 5, stride=2, pad=1)
            self.conv2 = L.Convolution2D(None, 256, 5, stride=1, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))

        return self.fc8(h)

# loading existing model
model = AlexNet()
serializers.load_npz('alex_cifar10.model', model)


# define pre_Alex
class AlexNet_cifar(chainer.Chain):

    def __init__(self):
        super(AlexNet_cifar, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 5, stride=2, pad=1, initialW=model.conv1.W.data,initial_bias=model.conv1.b.data)
            self.conv2 = L.Convolution2D(None, 256, 5, stride=1, pad=2, initialW=model.conv2.W.data,initial_bias=model.conv2.b.data)
            self.conv3 = L.Convolution2D(None, 384, 3, stride=1, pad=1, initialW=model.conv3.W.data,initial_bias=model.conv3.b.data)
            self.conv4 = L.Convolution2D(None, 384, 3, stride=1, pad=1, initialW=model.conv4.W.data,initial_bias=model.conv4.b.data)
            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1, initialW=model.conv5.W.data,initial_bias=model.conv5.b.data)
            self.fc6 = L.Linear(None, 4096, initialW=model.fc6.W.data,initial_bias=model.fc6.b.data)
            self.fc7 = L.Linear(None, 4096, initialW=model.fc7.W.data,initial_bias=model.fc7.b.data)
            self.fc8 = L.Linear(None, 10, initialW=model.fc8.W.data,initial_bias=model.fc8.b.data)

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)        
        h = F.local_response_normalization(h)
        h = self.conv2(h)        
        h = F.relu(h)        
        h = F.max_pooling_2d(h, 3, stride=2)        
        h = F.local_response_normalization(h)
        h = self.conv3(h)        
        h = F.relu(h)        
        h = self.conv4(h)       
        h = F.relu(h)    
        h = self.conv5(h)        
        h = F.relu(h)       
        h = F.max_pooling_2d(h, 3, stride=2)     
        h = self.fc6(h)    
        h = F.relu(h)    
        h = F.dropout(h)
        h = self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h)
        begin = datetime.datetime.now()
        h = self.fc8(h)
        end  = datetime.datetime.now()
        print((end - begin).total_seconds() * 1000, "ms")        
        return x
if __name__ == '__main__':
    alex_cifar10 = AlexNet_cifar()
    serializers.save_npz('part.model', alex_cifar10)    
    
    from _tool import chainerDataset
    _, _, x_test, y_test = chainerDataset.get_chainer_cifar10()
    # print(x_test[0].shape)
    output = alex_cifar10(x_test[0].reshape(1,3,32, 32))
    import cPickle as pickle
    pickle.dump(output, open('output.txt','wb'))
