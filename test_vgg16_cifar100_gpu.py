from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Load Network

import dill
with open("_models/train_vgg16_cifar100_gpu_(network).bn", "r") as f:
    branchyNet = dill.load(f)

# branchyNet.print_models()
print 'Load Network'


# Import Data

# from datasets import pcifar10
# _, _, x_test, y_test = pcifar10.get_data()

from _tool import chainerDataset
_, _, x_test, y_test = chainerDataset.get_chainer_cifar100()

print 'Import Data'


# Settings

TEST_BATCHSIZE = 1


print 'set network to inference mode'

branchyNet.to_gpu()
branchyNet.testing()
branchyNet.verbose = False
branchyNet.gpu = True


# Test main

g_baseacc, g_basediff, g_num_exits, g_accbreakdowns = utils.test(branchyNet, x_test, y_test, main=True,
                                                                 batchsize=TEST_BATCHSIZE)

print 'main accuracy: ', g_baseacc

g_basediff = (g_basediff / float(len(y_test))) * 1000.

branchyNet.to_cpu()
with open("_models/test_vgg16_cifar100_gpu_(g_baseacc).pkl", "w") as f:
    dill.dump({'g_baseacc': g_baseacc}, f)
with open("_models/test_vgg16_cifar100_gpu_(g_basediff).pkl", "w") as f:
    dill.dump({'g_basediff': g_basediff}, f)
with open("_models/test_vgg16_cifar100_gpu_(g_num_exits).pkl", "w") as f:
    dill.dump({'g_num_exits': g_num_exits}, f)
with open("_models/test_vgg16_cifar100_gpu_(g_accbreakdowns).pkl", "w") as f:
    dill.dump({'g_accbreakdowns': g_accbreakdowns}, f)

branchyNet.verbose = False

# Test branch
branchyNet.to_gpu()

b_baseacc, b_basediff, b_num_exits, b_accbreakdowns = utils.test(branchyNet, x_test, y_test,
                                                                 batchsize=TEST_BATCHSIZE)

branchyNet.to_cpu()
with open("_models/test_vgg16_cifar100_gpu_(b_baseacc).pkl", "w") as f:
    dill.dump({'b_baseacc': b_baseacc}, f)
with open("_models/test_vgg16_cifar100_gpu_(b_basediff).pkl", "w") as f:
    dill.dump({'b_basediff': b_basediff}, f)
with open("_models/test_vgg16_cifar100_gpu_(b_num_exits).pkl", "w") as f:
    dill.dump({'b_num_exits': b_num_exits}, f)
with open("_models/test_vgg16_cifar100_gpu_(b_accbreakdowns).pkl", "w") as f:
    dill.dump({'b_accbreakdowns': b_accbreakdowns}, f)

print 'b_baseacc: ', b_baseacc
print 'b_basediff: ', b_basediff
print 'b_num_exits: ', b_num_exits
print 'b_accbreakdowns: ', b_accbreakdowns

# Specify thresholds

# thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1., 5., 10.]
thresholds = [1.]  # 0.01  0.1  1  10 100

print 'thresholds: ', thresholds

print 'utils.screen_branchy()'

branchyNet.to_gpu()

branchyNet.gpu = True
branchyNet.verbose = False

g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                      batchsize=TEST_BATCHSIZE, verbose=True)

# convert to ms
g_diffs *= 1000.


branchyNet.to_cpu()
with open("_models/test_vgg16_cifar100_results_GPU_(g_ts).pkl", "w") as f:
    dill.dump({'g_ts': g_ts}, f)
with open("_models/test_vgg16_cifar100_results_GPU_(g_accs).pkl", "w") as f:
    dill.dump({'g_accs': g_accs}, f)
with open("_models/test_vgg16_cifar100_results_GPU_(g_diffs).pkl", "w") as f:
    dill.dump({'g_diffs': g_diffs}, f)
with open("_models/test_vgg16_cifar100_results_GPU_(g_exits).pkl", "w") as f:
    dill.dump({'g_exits': g_exits}, f)

print 'g_ts: ', g_ts
print 'g_accs: ', g_accs
print 'g_diffs: ', g_diffs
print 'g_exits: ', g_exits
