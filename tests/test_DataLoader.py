import pytest
import torch

from data.cifar_loader import CIFAR100

def test_data():
    cifar = CIFAR100() # Default download folder is #root#/assets/cifar-100

    train_loader = cifar.train_loader

    train_iter = iter(train_loader)
    train_imgs, train_labels = next(train_iter)

    assert train_imgs.shape == torch.Size([64, 3, 32, 32])
    assert train_labels.shape == torch.Size([64])
