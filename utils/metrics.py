import torch


def calculate_mae(pred, target):
    """计算平均绝对误差 MAE"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_mse(pred, target):
    """计算均方误差 MSE"""
    return torch.mean((pred - target) ** 2).item()
