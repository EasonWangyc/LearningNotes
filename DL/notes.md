# Deep Learning (深度学习)

深度学习是机器学习的一个分支，它用多层神经网络从数据中学习表示。传统机器学习通常依赖人工设计特征，而深度学习更强调让模型自己从原始输入中逐层提取特征。

从学习顺序上看，深度学习可以理解为：

```text
张量与线性变换
  -> 神经元和激活函数
  -> 损失函数和反向传播
  -> 多层感知机
  -> 卷积神经网络
  -> 序列模型
  -> Attention 和 Transformer
  -> 正则化、优化器、训练流程
```

## 深度学习代码中的基本名词

- **张量（tensor）**：带有一个或多个维度的数值容器。标量是 0 维张量，向量是 1 维张量，图像批次通常是 4 维张量。
- **批次（batch）**：一次送入模型的一组样本，第一维通常是 `batch_size`。
- **层（layer）**：接收输入张量并产生输出张量的计算模块，例如 `Linear`、`Conv2d` 和 `LayerNorm`。
- **参数（parameter）**：层中需要通过训练更新的权重和偏置。
- **激活函数（activation）**：施加在中间结果上的非线性函数，例如 ReLU、Sigmoid 和 Tanh。
- **前向传播（forward propagation）**：从输入开始，依次执行各层并计算输出的过程。
- **反向传播（backpropagation）**：利用链式法则从损失开始反向计算各参数梯度的过程。
- **自动求导（autograd）**：框架根据计算图自动记录运算并计算梯度的机制。

阅读下面的代码时，先看每个张量的形状变化，再看数据经过了哪些层，最后看输出如何参与损失和参数更新。

## 张量与神经网络输入

神经网络中的数据通常用张量表示。张量可以是标量、向量、矩阵，也可以是更高维的数据。

常见形状：

| 数据类型 | 常见形状 | 含义 |
|---|---|---|
| 表格样本 | `[batch_size, feature_dim]` | 每行是一个样本 |
| 灰度图像 | `[batch_size, 1, height, width]` | 1 个通道 |
| 彩色图像 | `[batch_size, 3, height, width]` | RGB 三通道 |
| 文本 token | `[batch_size, seq_len]` | 每个位置是 token id |
| 序列特征 | `[batch_size, seq_len, hidden_dim]` | 每个 token 有一个向量 |

```python
# ===== 创建不同任务对应的输入张量 =====
import torch

# 表格数据：4 个样本，每个样本有 10 个特征。
x_table = torch.randn(4, 10)
# 图像数据：批次、通道、高度、宽度，PyTorch 默认使用 NCHW 排布。
x_image = torch.randn(4, 3, 32, 32)
# 文本输入通常先保存 token id，形状是批次和序列长度。
x_tokens = torch.randint(0, 1000, (4, 16))

print("table:", x_table.shape)
print("image:", x_image.shape)
print("tokens:", x_tokens.shape)
```

### 线性层

线性层是神经网络中最基础的计算模块：

$$
y = xW^T + b
$$

其中 $x$ 是输入，$W$ 是权重，$b$ 是偏置。在线性层中，每一个输出维度都是输入特征的加权和。

```python
# ===== 验证 Linear 的矩阵计算 =====
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0, 3.0]])
linear = nn.Linear(in_features=3, out_features=2)

with torch.no_grad():
    # 固定权重和偏置，便于把模块输出与手写公式对照。
    linear.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))
    linear.bias.copy_(torch.tensor([0.0, 1.0]))

y = linear(x)
# Linear 内部执行 x @ weight.T + bias。
manual = x @ linear.weight.T + linear.bias

print("linear output:", y)
print("manual output:", manual)
```

### 神经元

一个神经元可以看作“线性变换 + 非线性激活”：

$$
a = f(w^Tx + b)
$$

如果没有非线性激活，多层线性层叠加后仍然等价于一个线性层。激活函数让网络可以拟合非线性关系。

```python
# ===== 对比三个常见激活函数的输出范围 =====
import torch

x = torch.linspace(-3, 3, steps=7)

# ReLU 将负数截断为 0；Sigmoid/Tanh 会把数值压缩到固定范围。
relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

print("x      :", x)
print("relu   :", relu)
print("sigmoid:", sigmoid.round(decimals=3))
print("tanh   :", tanh.round(decimals=3))
```

## 前向传播

前向传播是数据从输入层经过隐藏层，最后得到输出的过程。

对于一个两层网络：

$$
h = f(xW_1^T + b_1)
$$

$$
\hat{y} = hW_2^T + b_2
$$

```python
# ===== 一个两层网络的前向传播 =====
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        # 第一层提取中间特征，ReLU 提供非线性，第二层产生分类分数。
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out

model = TinyMLP()
x = torch.randn(5, 4)
# 输出形状 [5, 3] 表示 5 个样本、每个样本 3 个类别分数。
logits = model(x)

print("input:", x.shape)
print("output:", logits.shape)
```

### Logits 与概率

分类模型的最后一层通常先输出 logits。logits 不是概率，可以是任意实数。通过 softmax 才能变成概率分布：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
# ===== 将分类 logits 转成概率分布 =====
import torch

logits = torch.tensor([[2.0, 1.0, 0.1]])
# dim=-1 表示沿类别维度归一化，输出每行概率和为 1。
prob = torch.softmax(logits, dim=-1)

print("logits:", logits)
print("prob:", prob)
print("sum:", prob.sum(dim=-1))
```

## 损失函数

损失函数用来衡量模型预测和真实标签之间的差距。训练神经网络的核心就是不断调整参数，让损失变小。

**回归**预测连续数值，常用 MSE；**分类**预测离散类别，常用交叉熵。训练时模型输出通常先进入损失函数，损失函数再为反向传播提供一个标量目标。

## 回归损失

回归任务常用均方误差：

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

```python
# ===== 对照 MSELoss 和手动计算的均方误差 =====
import torch
import torch.nn as nn

y_true = torch.tensor([[1.0], [2.0], [3.0]])
y_pred = torch.tensor([[1.2], [1.8], [3.5]])

loss_fn = nn.MSELoss()
# loss_fn 将所有元素的平方误差求平均，得到一个标量张量。
loss = loss_fn(y_pred, y_true)

manual = ((y_pred - y_true) ** 2).mean()
print("mse:", loss.item())
print("manual:", manual.item())
```

## 分类损失

多分类任务常用交叉熵损失。PyTorch 的 `nn.CrossEntropyLoss` 接收的是 logits，而不是 softmax 后的概率。

它内部等价于：

```text
logits -> log_softmax -> negative log likelihood
```

```python
# ===== CrossEntropyLoss 直接接收 logits 和类别编号 =====
import torch
import torch.nn as nn

logits = torch.tensor([
    [2.0, 0.5, 0.1],
    [0.2, 1.5, 0.3],
])
target = torch.tensor([0, 1])

loss_fn = nn.CrossEntropyLoss()
# target 中的 0、1 是类别索引，不是 one-hot 向量。
loss = loss_fn(logits, target)

# 手动展开内部步骤，用于理解“log_softmax + 负对数似然”。
log_prob = torch.log_softmax(logits, dim=-1)
manual = -torch.stack([log_prob[0, 0], log_prob[1, 1]]).mean()

print("cross entropy:", loss.item())
print("manual:", manual.item())
```

## 反向传播

反向传播利用链式法则，计算损失函数对每个参数的梯度。

**梯度**是损失对参数的偏导数，表示参数发生微小变化时损失如何变化。`.backward()` 只负责计算梯度，不负责修改参数；参数更新通常由优化器的 `.step()` 完成。

对于简单函数：

$$
y = wx + b
$$

$$
L = (y - t)^2
$$

可以手动算出：

$$
\frac{\partial L}{\partial w} = 2(y-t)x
$$

$$
\frac{\partial L}{\partial b} = 2(y-t)
$$

```python
# ===== 用 autograd 验证线性模型的梯度 =====
import torch

x = torch.tensor(3.0)
t = torch.tensor(10.0)

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# requires_grad=True 让 PyTorch 记录后续运算，并构建计算图。
y = w * x + b
loss = (y - t) ** 2
# backward 沿计算图反向传播，把梯度写入 w.grad 和 b.grad。
loss.backward()

print("y:", y.item())
print("loss:", loss.item())
print("grad w:", w.grad.item())
print("grad b:", b.grad.item())
```

### 计算图

PyTorch 会在前向传播时动态构建计算图。只要张量设置了 `requires_grad=True`，它参与的运算就会被记录下来。

调用 `loss.backward()` 后，梯度会累积到叶子张量的 `.grad` 上。

```python
# ===== 查看一个简单表达式对应的计算图结果 =====
import torch

x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

y = (x * w).sum()
z = y ** 2
# z 是最终标量，反向传播会沿 z -> y -> w 的路径应用链式法则。
z.backward()

print("y:", y.item())
print("z:", z.item())
print("w.grad:", w.grad)
```

### 梯度清零

PyTorch 默认会累积梯度。每次参数更新前通常需要调用 `optimizer.zero_grad()`。

```python
# ===== 观察梯度累积和清零 =====
import torch

w = torch.tensor(1.0, requires_grad=True)

loss1 = (w - 2) ** 2
loss1.backward()
print("after first backward:", w.grad.item())

loss2 = (w - 3) ** 2
# 第二次 backward 会把新梯度加到旧梯度上，而不是覆盖旧值。
loss2.backward()
print("after second backward:", w.grad.item())

w.grad.zero_()
# 清零后下一轮 backward 才会从干净的梯度开始。
print("after zero:", w.grad.item())
```

# 多层感知机

多层感知机（MLP）由多个线性层和非线性激活函数组成，适合处理表格数据、向量特征，也经常作为其他模型中的分类头或投影层。

## MLP 的结构

一个简单 MLP 可以写成：

```text
input -> Linear -> ReLU -> Linear -> output
```

```python
# ===== 定义一个可复用的 MLP 模块 =====
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # Sequential 按声明顺序执行 Linear -> ReLU -> Linear。
        return self.net(x)

model = MLP()
x = torch.randn(8, 2)
logits = model(x)

print(logits.shape)
```

## 用 MLP 拟合非线性数据

下面构造一个二维二分类任务：如果点在单位圆内，则为 1；否则为 0。这个决策边界是非线性的，单层线性分类器不容易表达，而 MLP 可以通过隐藏层拟合这种边界。

```python
# ===== 用 MLP 拟合圆形的非线性决策边界 =====
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.rand(512, 2) * 4 - 2
# 圆内标签为 1，圆外标签为 0；这不是直线可以直接分开的数据。
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).long()

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for step in range(80):
    # 前向传播得到每个样本的两个类别 logits。
    logits = model(X)
    loss = loss_fn(logits, y)

    # 一个训练步依次为：清零旧梯度、反向传播、更新参数。
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred = model(X).argmax(dim=-1)
    acc = (pred == y).float().mean()

print("loss:", round(loss.item(), 4))
print("accuracy:", round(acc.item(), 3))
```

# 卷积神经网络

卷积神经网络（CNN）主要用于图像、视频、二维网格信号等数据。它的核心思想是局部感受野和权重共享。

## 卷积

卷积层用一个小的卷积核在图像上滑动，每个位置计算局部区域和卷积核的加权和。

**卷积核（kernel/filter）**是一组可学习的小权重；**通道（channel）**表示输入或输出特征图的数量；**感受野（receptive field）**表示一个输出位置能够看到的输入区域。

对于图像来说，卷积可以提取边缘、纹理、角点等局部模式。

```python
# ===== 用固定卷积核观察局部差异 =====
import torch
import torch.nn as nn

x = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
# 输入形状为 [batch=1, channel=1, height=3, width=3]。
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)

with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]]))

y = conv(x)
# kernel_size=2 且未设置 padding，空间尺寸从 3x3 变为 2x2。

print("input:")
print(x[0, 0])
print("kernel:")
print(conv.weight[0, 0])
print("output:")
print(y[0, 0])
```

### Padding 与 Stride

卷积输出空间大小由输入大小、卷积核大小、padding 和 stride 决定：

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1
$$

```python
# ===== 对比 stride=1 和 stride=2 的空间尺寸 =====
import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)

conv_same = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
conv_down = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)

# stride=2 每次移动两格，因此输出空间尺寸约缩小一半。
print("same:", conv_same(x).shape)
print("down:", conv_down(x).shape)
```

## 池化

池化用于降低特征图尺寸。MaxPool 取局部区域最大值，AvgPool 取局部区域平均值。

```python
# ===== 对比最大池化和平均池化 =====
import torch
import torch.nn as nn

x = torch.tensor([[[[
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
]]]]).view(1, 1, 4, 4)

max_pool = nn.MaxPool2d(kernel_size=2)
avg_pool = nn.AvgPool2d(kernel_size=2)

# 两种池化都把每个 2x2 区域压缩为一个值，但聚合规则不同。
print("max pool:")
print(max_pool(x)[0, 0])
print("avg pool:")
print(avg_pool(x)[0, 0])
```

## BatchNorm

BatchNorm 会在 batch 维度上统计均值和方差，并对激活进行归一化：

$$
\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

训练时 BatchNorm 使用当前 batch 的统计量，推理时使用训练过程中累计的 running mean 和 running variance。

**归一化（normalization）**是把中间激活调整到较稳定的数值范围。BatchNorm 的统计维度与 batch 和通道有关，而 LayerNorm 通常在每个样本的特征维度上计算统计量；二者适合的网络结构和 batch 条件不同。

```python
# ===== 观察 BatchNorm 对每个特征列的标准化效果 =====
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(3)
x = torch.tensor([
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [4.0, 40.0, 400.0],
])

y = bn(x)
# 默认处于 train 模式，因此这里使用当前 batch 的均值和方差。

print("mean:", y.mean(dim=0).round(decimals=5))
print("std:", y.std(dim=0, unbiased=False).round(decimals=5))
```

## 一个简单 CNN

```python
# ===== 组合卷积、归一化、激活和池化 =====
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # features 将 28x28 图像下采样到 7x7，再交给分类头。
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN()
x = torch.randn(4, 1, 28, 28)
# 两次 2 倍池化后，空间尺寸为 7x7，通道数为 32。
logits = model(x)

print("logits:", logits.shape)
print("parameters:", sum(p.numel() for p in model.parameters()))
```

## 残差连接

深层网络训练困难的一个原因是梯度需要穿过很多层。残差连接让网络学习一个残差函数：

$$
y = F(x) + x
$$

这样即使 $F(x)$ 暂时学得不好，信息仍然可以通过恒等路径向后传播。

**残差（residual）**是目标映射与输入之间的差值。代码中的 `self.block(x) + x` 要求两条路径形状一致；如果通道数不同，需要额外的投影层调整形状。

```python
# ===== 验证残差分支与恒等分支相加 =====
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        # 主分支学习残差，跳连分支保留原始输入。
        return self.act(self.block(x) + x)

x = torch.randn(2, 16, 32, 32)
block = ResidualBlock(16)
y = block(x)

print(y.shape)
```

# 序列模型

序列模型处理的数据有时间或顺序关系，例如文本、语音、轨迹、传感器时间序列。

## RNN

RNN 在每个时间步接收当前输入 $x_t$ 和上一时刻隐状态 $h_{t-1}$，输出新的隐状态：

$$
h_t = \tanh(W_xx_t + W_hh_{t-1} + b)
$$

隐状态可以理解为模型到当前时刻为止的记忆。

**时间步（time step）**是序列中的一个位置；**隐状态（hidden state）**是模型在该位置携带的中间表示。`batch_first=True` 时，输入形状为 `[batch, seq_len, input_size]`。

```python
# ===== 查看 RNN 输出和最终隐状态的形状 =====
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=5, hidden_size=8, batch_first=True)
x = torch.randn(4, 6, 5)

out, h_n = rnn(x)
# out 保存每个时间步的隐状态，h_n 保存最后时间步的状态。

print("out:", out.shape)
print("h_n:", h_n.shape)
```

### 手写单步 RNN

```python
# ===== 手动实现一个时间步的 RNN 更新 =====
import torch

batch, input_dim, hidden_dim = 2, 3, 4
x_t = torch.randn(batch, input_dim)
h_prev = torch.zeros(batch, hidden_dim)

W_x = torch.randn(input_dim, hidden_dim)
W_h = torch.randn(hidden_dim, hidden_dim)
b = torch.zeros(hidden_dim)

# 当前输入和上一时刻状态分别经过线性变换后相加，再施加 Tanh。
h_t = torch.tanh(x_t @ W_x + h_prev @ W_h + b)

print(h_t.shape)
```

## LSTM

普通 RNN 容易在长序列上出现梯度消失或爆炸。LSTM 引入细胞状态和门控机制，让信息可以更稳定地跨时间传递。

LSTM 的三个核心门：

| 门 | 作用 |
|---|---|
| 遗忘门 | 决定旧细胞状态保留多少 |
| 输入门 | 决定新信息写入多少 |
| 输出门 | 决定当前隐状态输出多少 |

**门控（gate）**是取值在 0 到 1 附近的控制信号，可以理解为对信息进行保留或过滤。LSTM 的细胞状态负责长期传递，隐状态负责当前时间步的输出。

```python
# ===== 观察双向、两层 LSTM 的输出形状 =====
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
)

x = torch.randn(4, 6, 5)
out, (h_n, c_n) = lstm(x)
# bidirectional=True 会把正向和反向隐状态拼接，因此最后一维变为 16。

print("out:", out.shape)
print("h_n:", h_n.shape)
print("c_n:", c_n.shape)
```

## GRU

GRU 可以看作 LSTM 的简化版本。它合并了一部分门控结构，参数更少，训练更快。

```python
import torch
import torch.nn as nn

gru = nn.GRU(input_size=5, hidden_size=8, batch_first=True)
x = torch.randn(4, 6, 5)

out, h_n = gru(x)

print("out:", out.shape)
print("h_n:", h_n.shape)
```

# Attention 与 Transformer

RNN 需要按时间步顺序处理序列，而 Attention 可以让序列中的每个位置直接关注其他位置。Transformer 完全基于 Attention 和前馈网络，适合并行训练。

## Self-Attention

Self-Attention 会从输入中生成 Query、Key、Value：

**Query（查询）**表示当前位置想寻找什么，**Key（键）**表示每个位置可被匹配的特征，**Value（值）**表示被聚合的内容。注意力权重由 Query 和 Key 的相似度决定，再用这些权重加权 Value。

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

然后计算注意力：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

```python
# ===== 手动拆解 Self-Attention =====
import torch

torch.manual_seed(42)

X = torch.randn(2, 4, 8)
W_q = torch.randn(8, 8)
W_k = torch.randn(8, 8)
W_v = torch.randn(8, 8)

Q = X @ W_q
K = X @ W_k
V = X @ W_v

# QK^T 得到每个 query 对所有 key 的匹配分数，除以 sqrt(d_k) 稳定数值范围。
scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
# softmax 将每行分数转换成和为 1 的注意力权重。
weights = torch.softmax(scores, dim=-1)
# 权重与 V 相乘，得到每个位置聚合后的表示。
out = weights @ V

print("scores:", scores.shape)
print("weights:", weights.shape)
print("out:", out.shape)
print("row sum:", weights[0, 0].sum())
```

### Mask

在自回归生成中，当前位置不能看到未来 token，因此需要因果 mask。

```python
import torch

seq_len = 5
scores = torch.zeros(seq_len, seq_len)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

masked_scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)

print(weights)
```

## Multi-Head Attention

多头注意力会把 hidden dimension 切分成多个 head，让不同 head 学习不同关系。

**head** 是一组独立的注意力子空间。`embed_dim` 必须能被 `num_heads` 整除，每个 head 的维度为 `embed_dim / num_heads`。

```python
# ===== 调用 PyTorch 的多头注意力模块 =====
import torch
import torch.nn as nn

attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
x = torch.randn(2, 5, 16)

out, weights = attn(x, x, x)
# Self-Attention 的 Q、K、V 都来自同一个输入 x。

print("out:", out.shape)
print("weights:", weights.shape)
```

## Transformer Block

一个典型 Transformer Block 包含：

```text
输入
  -> LayerNorm
  -> Multi-Head Attention
  -> 残差连接
  -> LayerNorm
  -> FFN
  -> 残差连接
```

```python
# ===== Transformer Block：注意力、残差和 FFN =====
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x):
        # 先归一化，再做注意力，并通过残差连接回到主路径。
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # 第二个子层是前馈网络（FFN），同样通过残差连接保留输入信息。
        x = x + self.ffn(self.norm2(x))
        return x

block = TransformerBlock()
x = torch.randn(2, 8, 32)
y = block(x)

print(y.shape)
```

# 正则化

正则化的目标是减少过拟合，让模型在新数据上表现更稳定。

## Dropout

Dropout 在训练时随机将部分激活置零，迫使网络不要过度依赖某些神经元。

推理时 Dropout 会关闭。

**正则化（regularization）**是主动限制模型有效复杂度的方法。Dropout 通过随机丢弃激活降低神经元之间的共同依赖，`train()` 和 `eval()` 会切换它的行为。

```python
# ===== 对比 Dropout 的训练态和推理态 =====
import torch
import torch.nn as nn

x = torch.ones(10)
dropout = nn.Dropout(p=0.5)

dropout.train()
# 训练态会随机将约一半元素置零，并对剩余元素做缩放。
print("train:", dropout(x))

dropout.eval()
# 推理态不再随机丢弃，输出保持稳定。
print("eval:", dropout(x))
```

## LayerNorm

LayerNorm 对每个样本自己的 hidden dimension 做归一化，不依赖 batch 维度，因此在 NLP 和 Transformer 中非常常见。

`LayerNorm(4)` 表示对最后一个长度为 4 的特征维度计算均值和方差；它与 BatchNorm 的主要区别是统计量不跨样本 batch 共享。

```python
# ===== 验证 LayerNorm 对最后一维做标准化 =====
import torch
import torch.nn as nn

x = torch.randn(2, 3, 4) * 10 + 5
norm = nn.LayerNorm(4)
y = norm(x)
# 每个样本、每个位置的最后一维均值约为 0，标准差约为 1。

print("mean:", y.mean(dim=-1).round(decimals=5))
print("std:", y.std(dim=-1, unbiased=False).round(decimals=5))
```

## 权重衰减

权重衰减通过惩罚过大的权重，让模型参数更平滑。AdamW 是深度学习中常用的带权重衰减优化器。

**权重衰减（weight decay）**是在参数更新时持续压缩参数大小的正则化方式。`weight_decay` 是优化器超参数，不等同于把 L2 项手动加到所有损失中的实现细节。

```python
# ===== 执行一个带权重衰减的优化步骤 =====
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(4, 2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

x = torch.randn(8, 4)
y = torch.randint(0, 2, (8,))

loss = nn.CrossEntropyLoss()(model(x), y)
# 训练步骤仍然是清零梯度、反向传播、更新参数。
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", round(loss.item(), 4))
```

# 优化器与训练流程

神经网络训练通常包含以下步骤：

```text
准备数据
  -> 前向传播
  -> 计算损失
  -> 梯度清零
  -> 反向传播
  -> 参数更新
  -> 重复多个 epoch
```

## SGD

SGD 沿着当前 batch 的梯度方向更新参数：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

动量会累积历史梯度方向，使更新更稳定。

**批量梯度（batch gradient）**是在一小批样本上计算的梯度；**动量（momentum）**保存历史更新方向，减少单次 batch 噪声造成的来回摆动。

```python
# ===== 执行一个 SGD + momentum 更新 =====
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

x = torch.randn(16, 2)
y = x[:, :1] * 2 - x[:, 1:] + 0.5

loss = nn.MSELoss()(model(x), y)
# optimizer.step() 会根据 loss.backward() 写入的梯度修改 model.parameters()。
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", round(loss.item(), 4))
```

## Adam

Adam 会维护梯度的一阶矩和二阶矩，相当于给每个参数自适应学习率。

AdamW 将权重衰减从梯度更新中解耦出来，是训练 Transformer 时更常见的选择。

**一阶矩**可以理解为梯度的指数滑动平均，**二阶矩**是梯度平方的指数滑动平均；Adam 根据二者为不同参数调整有效步长。

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(32, 4)
y = torch.randint(0, 2, (32,))

for _ in range(5):
    # 每次循环都是一次完整的参数更新。
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("loss:", round(loss.item(), 4))
```

## 学习率调度

学习率过大可能震荡，过小训练太慢。学习率调度器会随着训练进程调整学习率。

**学习率调度器（scheduler）**根据 epoch、step 或预设曲线修改优化器中的学习率。这里的余弦调度会让学习率按余弦曲线逐步变化。

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(2, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(5):
    x = torch.randn(16, 2)
    y = x[:, :1] - 2 * x[:, 1:]

    loss = nn.MSELoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler 通常在参数更新后推进到下一个学习率。
    scheduler.step()

    print(epoch, "lr:", round(scheduler.get_last_lr()[0], 6))
```

## 一个完整训练循环

下面用随机生成的数据演示完整训练流程。这个例子不依赖下载数据集，适合检查训练循环结构。

```python
# ===== DataLoader 驱动的完整训练循环 =====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

X = torch.randn(512, 10)
true_w = torch.randn(10, 3)
y = (X @ true_w).argmax(dim=-1)

dataset = TensorDataset(X, y)
# DataLoader 负责按 batch 取数据，并在每个 epoch 中打乱样本顺序。
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(5):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        # xb/yb 是当前 batch 的输入和标签。
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 用样本数加权累计，最后得到整个 epoch 的平均损失和准确率。
        total_loss += loss.item() * len(xb)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total_count += len(xb)

    print(
        "epoch:", epoch,
        "loss:", round(total_loss / total_count, 4),
        "acc:", round(total_correct / total_count, 3),
    )
```

# 现代模型结构

## ResNet

ResNet 的核心贡献是残差连接。它不是简单堆叠卷积层，而是让每个 block 学习：

$$
F(x) = H(x) - x
$$

最后输出：

$$
H(x) = F(x) + x
$$

这让非常深的 CNN 更容易训练。

## ViT

Vision Transformer 将图像切成 patch，再把每个 patch 当成一个 token 输入 Transformer。

```text
image
  -> patchify
  -> linear projection
  -> add position embedding
  -> Transformer Encoder
  -> classification head
```

下面只演示 patchify 的张量形状变化：

```python
import torch

images = torch.randn(2, 3, 32, 32)
patch_size = 8

patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
patches = patches.permute(0, 2, 3, 1, 4, 5)
patches = patches.reshape(2, -1, 3 * patch_size * patch_size)

print("patches:", patches.shape)
```

## BERT 与 GPT

BERT 和 GPT 都基于 Transformer，但训练目标和注意力方式不同。

| 模型 | 注意力方式 | 典型训练目标 |
|---|---|---|
| BERT | 双向 Attention | Masked Language Modeling |
| GPT | 因果 Attention | Next Token Prediction |

在深度学习基础阶段，不需要先记住所有模型细节，更重要的是理解它们共享的基础模块：embedding、attention、MLP、residual、normalization 和 loss。

# 学习顺序

如果从零开始学习深度学习，可以按下面顺序推进：

1. 先熟悉张量形状、线性层、激活函数和损失函数。
2. 手写几个最小反向传播例子，理解梯度是怎么来的。
3. 用 MLP 跑通一个小型分类任务，掌握训练循环。
4. 学 CNN 的卷积、池化、BatchNorm 和残差连接。
5. 学 RNN、LSTM、GRU，建立序列建模直觉。
6. 学 Self-Attention 和 Transformer Block，为后续 LLM/VLM 做准备。
7. 学 Dropout、LayerNorm、AdamW、学习率调度，把训练流程写稳定。
