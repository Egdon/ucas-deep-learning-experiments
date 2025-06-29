import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class CIFAR10Dataset:
    """CIFAR-10数据集加载器"""
    
    def __init__(self, data_root="data/CIFAR-10", batch_size=256, num_workers=4):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # CIFAR-10类别名称
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 数据增强和预处理（优化版本）
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 减少crop范围，降低CPU负载
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 减少增强强度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self):
        """获取训练和验证数据加载器"""
        
        # 训练集
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_root, 'train'),
            transform=self.train_transform
        )
        
        # 验证集
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_root, 'val'),
            transform=self.val_transform
        )
        
        # 数据加载器（优化版本）
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # 保持worker进程，减少启动开销
            prefetch_factor=2  # 预取数据，提高效率
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader
    
    def get_dataset_info(self):
        """获取数据集信息"""
        train_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'train'))
        val_dataset = datasets.ImageFolder(os.path.join(self.data_root, 'val'))
        
        info = {
            'num_classes': len(self.classes),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'class_names': self.classes
        }
        
        return info

def create_data_loaders(data_root="data/CIFAR-10", batch_size=256, num_workers=4):
    """便捷函数：创建数据加载器"""
    dataset = CIFAR10Dataset(data_root, batch_size, num_workers)
    return dataset.get_dataloaders(), dataset.get_dataset_info() 