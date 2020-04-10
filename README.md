# Fine-grained Elastic Partitioning for Distributed DNN
    
## Requirements

- A machine with a decent GPU (CUDA10.1-cudnn7-devel)

- Python 2.7 (Anaconda python 2.7.15)

## Python Dependencies

- chainer = 1.24.0

- matplotlib

- dill = 0.2.5

- scikit-image

- scipy

## Quick Start

```
./train.sh & ./test.sh
```

or 

```
./experiment.sh
```

## Notice

For layer-level inference latency and output size analysiz, you need to update the chainer version from 1.24.0 to 2.0.2.
