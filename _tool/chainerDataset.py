import numpy
from chainer import datasets
import random

def get_lenet_mnist():
    raw_train, raw_test = datasets.get_mnist(withlabel=True, ndim=1, scale=1.)
    return process_data(raw_train, raw_test)


def get_chainer_cifar10():
    raw_train, raw_test = datasets.get_cifar10(withlabel=True, ndim=3, scale=1.)
    return process_data(raw_train, raw_test)


def get_chainer_cifar100():
    raw_train, raw_test = datasets.get_cifar100(withlabel=True, ndim=3, scale=1.)
    return process_data(raw_train, raw_test)


def process_data(raw_train, raw_test):
    list_raw_train_x = []
    list_raw_train_y = []
    list_raw_test_x = []
    list_raw_test_y = []

    for item_train in raw_train:
        list_raw_train_x.append(item_train[0])
        list_raw_train_y.append(item_train[1])

    for item_test in raw_test:
        list_raw_test_x.append(item_test[0])
        list_raw_test_y.append(item_test[1])

    x_train = numpy.array(list_raw_train_x)
    y_train = numpy.array(list_raw_train_y)
    x_test = numpy.array(list_raw_test_x)
    y_test = numpy.array(list_raw_test_y)

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape

    return x_train, y_train, x_test, y_test
