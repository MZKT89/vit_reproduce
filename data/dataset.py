# data/dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configs.config import DataConfig

data_config = DataConfig()

def get_transforms():
    transform = transforms.Compose([
        # 将PIL图像转为PyTorch Tensor
        # [0,255] -> [0,1]
        transforms.ToTensor(),
        # 归一化
        transforms.Normalize(
            mean=data_config.mean,
            std=data_config.std
        )
    ])
    return transform

def get_datasets():
    """加载Fashion-MNIST训练集和测试集"""
    transform = get_transforms()
    
    train_dataset = datasets.FashionMNIST(
        root=data_config.data_dir,  # ./data
        train=True,
        download=True,  
        transform=transform  
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_config.data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"训练集：{len(train_dataset)} 张图像")  
    print(f"测试集：{len(test_dataset)} 张图像")    
    return train_dataset, test_dataset

def get_dataloaders():
    """创建DataLoader（批量加载数据）"""
    train_dataset, test_dataset = get_datasets()
    
    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data_config.batch_size,  # batch size 64
        shuffle=True,  
        num_workers=data_config.num_workers, 
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,  #测试集不打乱
        num_workers=data_config.num_workers,
    )
    
    return train_loader, test_loader

def download_dataset():
    train_loader, test_loader = get_dataloaders()
    
    for images, labels in train_loader:
        print(f"图像批次形状：{images.shape}")  # (64, 1, 28, 28) 灰度图通道为1
        print(f"标签批次形状：{labels.shape}")  # (64,)
        print(f"图像像素值范围：[{images.min():.2f}, {images.max():.2f}]")  # [-1, 1]
        break  