import torch
from torch.utils.data import DataLoader
from data.MovingMNIST.MovingMNIST import MovingMNIST


def get_dataloaders(batch_size=16, num_workers=4, data_root='.data/mnist'):
    """
    获取 Moving MNIST 数据加载器

    Args:
        batch_size: 批次大小
        num_workers: 数据加载线程数
        data_root: 数据集根目录

    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """

    # 创建训练集
    train_dataset = MovingMNIST(
        root=data_root,
        train=True,
        download=True
    )

    # 创建测试集
    test_dataset = MovingMNIST(
        root=data_root,
        train=False,
        download=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
