# LZU Li Group AI Code

基于 Moving MNIST 数据集


### 训练模型
```bash
python main.py --visualize-only False --batch-size 32 --epochs 50
```

### 可视化已训练模型
```bash
python main.py --num-samples 2
```

## 如何添加模型

### 修改现有模型

在 `main.py` 中找到模型创建位置：

```python
# ========== create model ==========
# 后续修改只需在此处更改模型定义即可, 默认的任务是[B,C_in,H,W] -> [B,C_out,H,W]

model = U_Net(input_channel=10, num_classes=10).to(device)

# ========================================================
```

直接替换为你的模型：
```python
model = MyModel(input_channel=10, output_channel=10).to(device)
```


2. 在 `main.py` 中导入并使用：

```python
from model.my_model import MyModel

# 在模型创建位置替换
model = MyModel(input_channel=10, output_channel=10).to(device)
```

## 模型要求

- **输入**: `[Batch, 10, Height, Width]` - 10帧灰度图像
- **输出**: `[Batch, 10, Height, Width]` - 后续10帧灰度图像

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

