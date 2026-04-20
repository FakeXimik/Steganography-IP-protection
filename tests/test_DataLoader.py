import pytest
import torch
from unittest.mock import patch
from torch.utils.data import Dataset
from data.cifar_loader import CIFAR100

# --- FIXTURES & FAKES ---

class DummyCIFAR100Dataset(Dataset):
    """
    A lightweight, fake dataset that perfectly mimics the behavior of the 
    real CIFAR-100 dataset without downloading anything.
    """
    def __init__(self, length=128):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generates a random fake image tensor: 3 channels, 32x32 pixels
        fake_image = torch.randn(3, 32, 32)
        # Generates a random fake label between 0 and 99
        fake_label = torch.randint(0, 100, (1,)).item()
        
        return fake_image, fake_label

# --- TESTS ---

@patch("torchvision.datasets.CIFAR100")
def test_cifar100_dataloader_mocked(mock_cifar_dataset):
    # Setup the mock: Every time CIFAR100 is instantiated, return our fake dataset
    mock_cifar_dataset.return_value = DummyCIFAR100Dataset(length=128)

    # Initialize your class. 
    # Because of the patch, download=True is safely ignored!
    cifar = CIFAR100(batch_size=64, num_workers=0)

    # Pull a batch from the train loader
    train_iter = iter(cifar.train_loader)
    train_imgs, train_labels = next(train_iter)

    # Verify the dimensions match what the Neural Network expects
    assert train_imgs.shape == torch.Size([64, 3, 32, 32])
    assert train_labels.shape == torch.Size([64])

    # Ensure the test loader works the exact same way
    test_iter = iter(cifar.test_loader)
    test_imgs, test_labels = next(test_iter)
    assert test_imgs.shape == torch.Size([64, 3, 32, 32])

    # Verify your class actually tried to load both Train and Test sets
    assert mock_cifar_dataset.call_count == 2
