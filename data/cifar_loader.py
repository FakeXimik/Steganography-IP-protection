import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from pathlib import Path

class CIFAR100:
    def __init__(
            self,
            data_root: Path = Path.cwd() / "Assets" / "cifar-100",
            transform: torchvision.transforms.Compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
            num_workers: int = 2,
            batch_size: int = 64
    ):
        self.data_root = data_root
        
        self.transform = transform
        self.train_set = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform
        )
        self.test_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=transform
        )
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)