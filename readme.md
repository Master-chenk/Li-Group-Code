# Moving MNIST Video Prediction

视频预测框架 - 基于 Moving MNIST 数据集

## 快速开始

### 训练模型
```bash
python main.py --visualize-only False --batch-size 32 --epochs 50
```

### 可视化已训练模型
```bash
python main.py --num-samples 2
```

## 如何添加/修改模型

### 方法一：修改现有模型（推荐）

在 `main.py` 中找到模型创建位置（约 Line 238）：

```python
# ========== create model ==========
# 后续修改只需在此处更改模型定义即可, 默认的任务是[B,C_in,H,W] -> [B,C_out,H,W]

model = U_Net(input_channel=10, num_classes=10).to(device)

# ========================================================
```

直接替换为你的模型：
```python
model = YourModel(input_channel=10, num_classes=10).to(device)
```

### 方法二：添加新模型文件

1. 在 `model/` 目录下创建新模型文件，例如 `my_model.py`：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_channel=10, num_classes=10):
        super(MyModel, self).__init__()
        # 定义你的模型结构
        self.conv1 = nn.Conv2d(input_channel, 64, 3, 1, 1)
        # ...
        self.conv_out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # x shape: [B, C_in, H, W]
        # 实现前向传播
        x = self.conv1(x)
        # ...
        return self.conv_out(x)  # [B, C_out, H, W]
```

2. 在 `main.py` 中导入并使用：

```python
from model.my_model import MyModel

# 在模型创建位置替换
model = MyModel(input_channel=10, num_classes=10).to(device)
```

## 模型要求

- **输入**: `[Batch, 10, Height, Width]` - 10帧灰度图像
- **输出**: `[Batch, 10, Height, Width]` - 预测的后续10帧

## 项目结构

```
.
├── main.py              # 主程序（训练+可视化）
├── model/
│   └── U_Net.py         # U-Net模型（示例）
├── data/
│   └── moving_mnist.py  # 数据加载器
├── utils/
│   └── metrics.py       # 评估指标 (MAE, MSE)
├── checkpoints/         # 模型保存目录
└── results/             # 可视化结果保存目录
```

## 命令行参数

```bash
--visualize-only      # 仅可视化模式（默认：True）
--batch-size          # 批次大小（默认：32）
--epochs              # 训练轮数（默认：50）
--lr                  # 学习率（默认：1e-3）
--num-samples         # 可视化样本数量（默认：2）
--no-visualize        # 训练后不自动可视化
```

## 示例

### 训练自定义模型
```bash
# 1. 在 model/ 下添加你的模型
# 2. 在 main.py 中修改模型导入和创建
# 3. 运行训练
python main.py --visualize-only False --epochs 100 --lr 0.001
```

### 调整可视化
```bash
# 显示3组预测结果
python main.py --num-samples 3
```
