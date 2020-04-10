#!/bin/bash

echo "Train Part"

echo "alex cifar10"
python -u train_alex_cifar10_gpu.py
echo "alex cifar100"
python -u train_alex_cifar100_gpu.py

echo "vgg11 cifar10"
python -u train_vgg11_cifar10_gpu.py
echo "vgg11 cifar100"
python -u train_vgg11_cifar100_gpu.py

echo "vgg16 cifar10"
python -u train_vgg16_cifar10_gpu.py
echo "vgg16 cifar100"
python -u train_vgg16_cifar100_gpu.py

echo "resnet32 cifar10"
python -u train_resnet32_cifar10_gpu.py
echo "resnet32 cifar100"
python -u train_resnet32_cifar100_gpu.py

echo "resnet56 cifar10"
python -u train_resnet56_cifar10_gpu.py
echo "resnet56 cifar100"
python -u train_resnet56_cifar100_gpu.py

echo "resnet110 cifar10"
python -u train_resnet110_cifar10_gpu.py
echo "resnet110 cifar100"
python -u train_resnet110_cifar100_gpu.py

echo "mobilenet cifar10"
python -u train_mobileNet_cifar10_gpu.py
echo "mobilenet cifar100"
python -u train_mobileNet_cifar100_gpu.py


echo "Test Part"

echo "alex cifar10"
python -u test_alex_cifar10_gpu.py
echo "alex cifar100"
python -u test_alex_cifar100_gpu.py

echo "vgg11 cifar10"
python -u test_vgg11_cifar10_gpu.py
echo "vgg11 cifar100"
python -u test_vgg11_cifar100_gpu.py

echo "vgg16 cifar10"
python -u test_vgg16_cifar10_gpu.py
echo "vgg16 cifar100"
python -u test_vgg16_cifar100_gpu.py

echo "resnet32 cifar10"
python -u test_resnet32_cifar10_gpu.py
echo "resnet32 cifar100"
python -u test_resnet32_cifar100_gpu.py

echo "resnet56 cifar10"
python -u test_resnet56_cifar10_gpu.py
echo "resnet56 cifar100"
python -u test_resnet56_cifar100_gpu.py

echo "resnet110 cifar10"
python -u train_resnet110_cifar10_gpu.py
echo "resnet110 cifar100"
python -u train_resnet110_cifar100_gpu.py

echo "mobilenet cifar10"
python -u test_mobileNet_cifar10_gpu.py
echo "mobilenet cifar100"
python -u test_mobileNet_cifar100_gpu.py
