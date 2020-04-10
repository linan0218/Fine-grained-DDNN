from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Define Network

from networks import vgg11_cifar
branchyNet = vgg11_cifar.get_network(n_class=100)

# branchyNet.print_models()
print 'Define Network'

branchyNet.to_gpu()
branchyNet.training()


# Import Data

# from datasets import pcifar10
# x_train, y_train, _, _ = pcifar10.get_data()

from _tool import chainerDataset
x_train, y_train, _, _ = chainerDataset.get_chainer_cifar100()

print 'Import Data'


# Settings

TRAIN_BATCHSIZE = 512
TRAIN_NUM_EPOCHS = 100

branchyNet.verbose = False
branchyNet.gpu = True


# Train Main Network

print 'Train Main Network'

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

# print 'main_loss : ', main_loss, ' | main_acc : ', main_acc, ' | main_time : ', main_time


# Train BranchyNet

print 'Train BranchyNet'

TRAIN_NUM_EPOCHS = 100

branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHS)

# print 'branch_loss : ', branch_loss, ' | branch_acc : ', branch_acc, ' | branch_time : ', branch_time


print 'Save model/data'

import dill
branchyNet.to_cpu()
with open("_models/train_vgg11_cifar100_gpu_(network).bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/train_vgg11_cifar100_gpu_(main_loss).pkl", "w") as f:
    dill.dump({'main_loss': main_loss}, f)
with open("_models/train_vgg11_cifar100_gpu_(main_acc).pkl", "w") as f:
    dill.dump({'main_acc': main_acc}, f)
with open("_models/train_vgg11_cifar100_gpu_(main_time).pkl", "w") as f:
    dill.dump({'main_time': main_time}, f)
with open("_models/train_vgg11_cifar100_gpu_(branch_loss).pkl", "w") as f:
    dill.dump({'branch_loss': branch_loss}, f)
with open("_models/train_vgg11_cifar100_gpu_(branch_acc).pkl", "w") as f:
    dill.dump({'branch_acc': branch_acc}, f)
with open("_models/train_vgg11_cifar100_gpu_(branch_time).pkl", "w") as f:
    dill.dump({'branch_time': branch_time}, f)
