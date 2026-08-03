# AI Infrastructure

> AI 基础设施学习笔记

AI Infrastructure（AI Infra）关注的是：如何为模型训练、微调、推理和数据处理提供稳定、可扩展、可观测的计算、存储、网络和软件系统。

```text
硬件资源：CPU、GPU/NPU、内存、显存、磁盘、网络和互联拓扑
软件栈：Driver、Toolkit、Runtime、算子库、通信库、框架和容器
工作负载：预训练、微调、评估、离线推理和在线推理
工程质量：性能、弹性、可观测性、资源利用率和安全
```

## 内容概览

- CPU、GPU、NPU 与异构计算
- FLOPS、带宽、算术强度和 Roofline 模型
- 显存、KV Cache、NUMA 与 GPU 拓扑
- CUDA 软件栈、容器和运行时兼容性
- 集合通信、NCCL 和分布式训练
- DP、TP、PP、FSDP、ZeRO、Sequence/Context Parallel
- 训练显存、激活检查点和 Offload
- Prefill、Decode、KV Cache、PagedAttention 和连续批处理
- FP16、BF16、FP8、INT8、GPTQ、AWQ 等量化
- 推理服务、监控、Profiling 和故障定位

## 1. AI Infra 的分层结构

### 1.1 硬件层

| 资源 | 主要作用 | 常见瓶颈 |
|------|----------|----------|
| CPU | 数据预处理、控制逻辑、调度和 I/O | 核数、主频、内存带宽 |
| GPU/NPU | 矩阵、向量和张量计算 | 计算吞吐、显存、片上带宽 |
| Host Memory | 数据集、缓存、Pinned Memory、CPU Offload | 容量、带宽、NUMA |
| Device Memory | 参数、激活、梯度、KV Cache | 容量、带宽、碎片 |
| 本地 NVMe | 数据集、Checkpoint 和临时缓存 | IOPS、顺序带宽、容量 |
| 节点网络 | 数据加载、Checkpoint 和跨节点通信 | 带宽、延迟、拥塞 |

### 1.2 软件层

```text
应用 / 训练脚本 / 推理服务
            │
框架：PyTorch、JAX、TensorFlow
            │
算子与通信库：cuBLAS、cuDNN、NCCL、Triton
            │
运行时：CUDA Runtime、Driver API
            │
设备驱动与操作系统
            │
GPU / NPU 硬件
```

### 1.3 工作负载层

| 工作负载 | 目标 | 典型指标 |
|----------|------|----------|
| 预训练 | 在大规模数据上优化模型参数 | tokens/s、MFU、训练时间 |
| 微调 | 使用较小数据集改变模型行为 | step time、显存、收敛速度 |
| 评估 | 运行固定数据集和指标 | 样本吞吐、准确率、可复现性 |
| 离线推理 | 批量处理大量输入 | 总吞吐、每样本成本 |
| 在线推理 | 响应交互请求 | TTFT、TPOT、P50/P95/P99 |

设计时先确认四件事：

1. 计算受算力限制，还是受显存/内存带宽限制？
2. 通信发生在单卡、单机多卡，还是跨节点？
3. 工作负载更重视吞吐，还是单请求延迟？
4. 故障、扩容、升级和回滚是否可以被观测和控制？

## 2. CPU、GPU 与 NPU

### 2.1 CPU

CPU 通常拥有较少但较复杂的核心，每个核心有较强的控制流处理能力：

- 分支预测和乱序执行能力强。
- 单线程延迟低。
- 适合复杂控制逻辑、系统调用、数据预处理和调度。
- 通常共享较大的 Cache 和系统内存。

### 2.2 GPU

GPU 使用大量计算单元执行相似指令：

- 适合大规模数据并行。
- 线程以 Warp 等硬件调度单位组织。
- 需要较高并行度来隐藏访存延迟。
- 对分支分化、随机访存和小规模任务较敏感。

矩阵乘、卷积、归一化和逐元素操作，通常具有规则的数据并行性，因此适合映射到 GPU。

### 2.3 NPU 与专用加速器

NPU、TPU、ASIC 等专用加速器通常针对矩阵乘、卷积、张量搬运或特定数据类型进行硬件设计：

- 能效可能优于通用 GPU。
- 算子覆盖、编译器和运行时更依赖具体平台。
- 不同厂商之间的模型格式、算子语义和调试方式可能不同。

比较 CPU/GPU/NPU 时，不应只看峰值 TOPS/TFLOPS，还要同时看：

- 可用精度和稀疏模式。
- 内存容量和带宽。
- Host/Device 传输开销。
- 编译器和运行时的成熟度。

## 3. 性能基础

### 3.1 FLOPS

FLOPS（Floating-point Operations Per Second）表示每秒浮点运算次数：

```text
1 GFLOPS = 10^9 FLOPS
1 TFLOPS = 10^12 FLOPS
1 PFLOPS = 10^15 FLOPS
```

峰值算力必须绑定数据类型和计算模式：

```text
峰值算力 = 数据类型 + 运算单元 + 稠密/稀疏模式 + 时钟条件
```

### 3.2 内存带宽

内存带宽描述单位时间内能够搬运的数据量：

$$
\text{Bandwidth} = \text{Transfer Rate} \times \text{Bus Width}
$$

| 带宽层次 | 位置 | 影响 |
|----------|------|------|
| Register | Thread 内部 | 局部变量和中间结果 |
| Shared Memory/L1 | SM 内 | Block 内数据复用 |
| L2 | GPU 内共享 Cache | 多个 SM 的缓存访问 |
| HBM/GDDR | GPU 显存 | 大量 Tensor 读写 |
| PCIe/NVLink | CPU/GPU 或 GPU/GPU | 节点内数据传输 |
| InfiniBand/RoCE | 节点间网络 | 分布式参数和梯度通信 |

### 3.3 算术强度

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}}
$$

- 向量加法读取两个元素并写入一个元素，通常偏 Memory-Bound。
- 大矩阵乘会重复使用 Tile 中的数据，通常更接近 Compute-Bound。
- LayerNorm 等操作常受带宽和 Kernel 融合影响。

### 3.4 Roofline 模型

$$
\text{Attainable Performance} = 
\min(\text{Peak Compute}, \text{Arithmetic Intensity} \times \text{Peak Bandwidth})
$$

```text
可达算力
   ▲
   │             ───────── 峰值算力上限
   │           /
   │         /
   │       /
   │     /
   └──────────────────────► 算术强度
       带宽受限区     算力受限区
```

使用步骤：

1. 估算或测量 Kernel 的 FLOPs。
2. 统计从显存层次搬运的字节数。
3. 计算算术强度。
4. 根据硬件带宽和算力定位瓶颈。
5. 决定优先优化数据复用、访存布局还是指令吞吐。

Roofline 是上界分析，不等于实际性能；启动、同步、分支、Cache、资源限制和通信都会进一步降低结果。

## 4. 内存、存储与互联

### 4.1 训练显存的主要组成

```text
训练显存
├── Parameters：模型参数
├── Gradients：反向传播梯度
├── Optimizer States：一阶、二阶矩和 Master Weights
├── Activations：反向传播需要的前向结果
├── Temporary Buffers：算子临时空间
├── Communication Buffers：集合通信缓冲区
└── CUDA Context / Allocator：运行时和分配器开销
```

显存估算需要明确参数、梯度和优化器状态的数据类型，还要考虑 Activation Checkpointing、ZeRO/FSDP、CPU Offload 和 NVMe Offload。

### 4.2 KV Cache 的显存估算

自回归推理中，每层的 Key/Value 会随着序列增长：

$$
\text{KV Bytes} \approx
2 \times L \times T \times B \times H_{kv} \times D \times S
$$

其中 `L` 是层数，`T` 是序列长度，`B` 是并发序列数，`H_kv` 是 KV Head 数，`D` 是 Head 维度，`S` 是元素字节数，前面的 `2` 表示 K 和 V。

GQA/MQA 通过减少 `H_kv` 降低 KV Cache 规模。长上下文、高并发和多轮会话会显著放大 KV Cache 压力。

### 4.3 NUMA

多路 CPU 系统中，不同 CPU Socket 对本地内存的访问延迟不同，这种结构称为 NUMA：

```text
NUMA Node 0                         NUMA Node 1
┌──────────────┐   互联             ┌──────────────┐
│ CPU 0        │◄──────────────────►│ CPU 1        │
│ Memory 0     │                    │ Memory 1     │
│ GPU 0/1      │                    │ GPU 2/3      │
└──────────────┘                    └──────────────┘
```

```bash
lscpu
numactl --hardware
numactl --show
nvidia-smi topo -m
```

常见优化方向：

- 让数据加载进程绑定到接近目标 GPU 的 CPU Node。
- 让网络中断和通信线程使用合理的 CPU 核。
- 使用 `numactl` 或调度器的 NUMA 约束。

### 4.4 GPU 拓扑

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
```

| 链路 | 特点 |
|------|------|
| PCIe | 通用、兼容性好，带宽和延迟受代际及拓扑影响 |
| NVLink | GPU 间高速互联，适合频繁 GPU-GPU 通信 |
| NVSwitch | 多 GPU 节点中的高速交换结构 |
| InfiniBand | 常用于高性能节点间通信 |
| RoCE | 基于 Ethernet 的 RDMA 方案，需要网络配置支持 |

GPU 编号不一定等于物理距离。分布式训练的 Rank 映射应结合拓扑，而不是只按设备编号排列。

### 4.5 通信时间的粗略模型

$$
T_{transfer} \approx \alpha + \frac{N}{\beta}
$$

其中 `α` 是启动延迟，`N` 是消息大小，`β` 是有效带宽。小消息更容易受延迟影响，大消息更容易受带宽影响。

## 5. GPU 软件栈和运行时

### 5.1 CUDA 软件栈

```text
应用：PyTorch / JAX / 推理服务
  │
算子库：cuBLAS / cuDNN / cuFFT / CUTLASS
通信库：NCCL
  │
CUDA Runtime API / Driver API
  │
NVIDIA Driver
  │
GPU 硬件
```

### 5.2 Driver、Toolkit 和 Runtime

| 组件 | 主要内容 | 典型检查 |
|------|----------|----------|
| Driver | 内核模块和用户态驱动接口 | `nvidia-smi` |
| Toolkit | `nvcc`、头文件、开发库和工具 | `nvcc --version` |
| Runtime | 应用调用的 CUDA API | 由程序和框架使用 |
| Framework Build | 框架构建时绑定的 CUDA 能力 | `torch.version.cuda` |

```bash
nvidia-smi
nvcc --version
ldconfig -p | grep -E 'cuda|cudnn|nccl'
```

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 5.3 容器中的 GPU 访问

容器通常携带用户态库和应用，但 GPU 设备访问依赖宿主机驱动与容器运行时配置：

```bash
docker run --rm --gpus all nvidia/cuda:runtime nvidia-smi
```

排查容器 GPU 问题时按层检查：

1. 宿主机 `nvidia-smi` 是否正常。
2. 容器运行时是否安装并启用 NVIDIA Container Toolkit。
3. 容器内是否能看到 `/dev/nvidia*` 设备。
4. 容器中的 CUDA 用户态库是否适合应用。
5. 框架是否识别设备和目标计算能力。

### 5.4 编译与运行时兼容性

需要区分三类兼容性：

- **驱动兼容性**：驱动是否支持应用所需的 CUDA Runtime。
- **二进制架构兼容性**：产物是否包含目标 GPU 的 `sm_XX` 或可用 PTX。
- **ABI/库兼容性**：C++ 编译器、框架、扩展和动态库的 ABI 是否一致。

建议保存完整环境信息：

```text
GPU 型号与数量
Driver Version
CUDA Toolkit Version
Framework Version
NCCL Version
Python Version
Container Image Digest
编译参数和环境变量
```

## 6. 集合通信

### 6.1 常见通信原语

| 原语 | 输入/输出关系 | 常见用途 |
|------|---------------|----------|
| Broadcast | 一个 Rank 发给所有 Rank | 同步初始参数 |
| Reduce | 所有 Rank 聚合到一个 Rank | 汇总梯度或统计量 |
| AllReduce | 聚合后所有 Rank 都得到结果 | DDP 梯度同步 |
| AllGather | 每个 Rank 的数据收集到所有 Rank | 参数/激活拼接 |
| ReduceScatter | 聚合后切分给各 Rank | ZeRO/FSDP 分片 |
| All-to-All | 每个 Rank 向所有 Rank 发送不同分片 | MoE Expert Parallel |

### 6.2 Ring AllReduce

Ring AllReduce 将通信拆成 Reduce-Scatter 和 AllGather：

```text
Reduce-Scatter：每个 Rank 接收并累加相邻 Rank 的分片
        │
        ▼
AllGather：每个 Rank 把最终分片传播给其他 Rank
        │
        ▼
所有 Rank 得到完整聚合结果
```

实际性能取决于链路拓扑、算法、消息大小、参与者数量和网络拥塞。

### 6.3 NCCL 的基本检查

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```

常见排查变量：

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

这些变量必须根据实际网卡和拓扑配置，不能机械复制。调试完成后应避免长期保留过高日志级别。

### 6.4 PyTorch 分布式最小结构

```python
import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

tensor = torch.ones(1, device="cuda") * (dist.get_rank() + 1)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print("rank:", dist.get_rank(), "value:", tensor.item())

dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=4 distributed_demo.py
```

### 6.5 通信瓶颈的表现

- GPU 利用率周期性下降，时间线中出现长通信区间。
- 每一步耗时随着 Rank 数增加明显增长。
- AllReduce 等待时间占比高。
- GPU 计算已经很快，但跨节点网络没有足够带宽。
- CPU 线程、网卡中断或 NUMA 绑定不合理。

## 7. 分布式训练并行方式

### 7.1 数据并行 DP

每个 Rank 保存一份模型副本，各自处理不同数据分片，再同步梯度：

```text
Rank 0：完整模型 + batch 0 ─┐
Rank 1：完整模型 + batch 1 ─┼─ AllReduce(gradients)
Rank 2：完整模型 + batch 2 ─┘
```

优点是实现简单、计算独立；限制是每张卡都需要存放完整模型和优化器状态。

### 7.2 DDP

PyTorch DDP 通常为每个进程绑定一张 GPU：

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

DDP 通过梯度 Bucket 在反向过程中进行通信，使部分通信可以与后续反向计算重叠。

### 7.3 张量并行 TP

张量并行把同一层的权重或激活切分到多个设备：

```text
Y = X × W

列切分：W = [W0 | W1]
Rank 0：Y0 = X × W0
Rank 1：Y1 = X × W1
结果：Y = [Y0 | Y1]
```

TP 每层都可能产生通信，因此通常适合高速 GPU-GPU 互联域。切分维度需要与矩阵布局、归一化和后续算子匹配。

### 7.4 流水线并行 PP

按层把模型切成多个 Stage，使用 Micro-batch 流经各 Stage：

```text
Stage 0 ──> Stage 1 ──> Stage 2 ──> Stage 3
   │           │           │           │
 Micro-batch 0, 1, 2, 3  依次流水执行
```

流水线存在 Pipeline Bubble。增加 Micro-batch 数量可以降低气泡比例，但会增加调度、激活和梯度管理复杂度。

### 7.5 3D 并行

```text
TP：切单层内部计算，通信频繁，适合高速互联
PP：切模型层，通信频率较低，产生流水线气泡
DP：复制并行副本，跨副本同步梯度
```

$$
N_{world} = N_{TP} \times N_{PP} \times N_{DP}
$$

选择映射时通常把 TP 放在最快互联域内，把 PP 放在通信更少的边界，把 DP 放到更大的网络范围。

### 7.6 Sequence Parallel 与 Context Parallel

Sequence Parallel 将序列维度或激活分布到多个设备，减少单卡激活和临时缓冲区；Context Parallel 进一步把长上下文处理分配到不同设备，通常需要额外的注意力通信。

它们和 TP/PP 的关系取决于具体框架实现：

```text
模型维度切分：Tensor Parallel
层间切分：Pipeline Parallel
Batch/副本切分：Data Parallel
序列维度切分：Sequence/Context Parallel
```

### 7.7 Expert Parallel

MoE 模型中的 Expert 可以分布在不同 GPU 上。Router 先为 Token 选择 Expert，再通过 All-to-All 把 Token 发送到对应设备：

```text
Token
  │ Router 选择 Expert
  ▼
All-to-All 发送到 Expert 所在 Rank
  │
  ▼
Expert 计算
  │
  ▼
All-to-All 把结果发回原 Rank
```

Expert Parallel 的主要瓶颈常来自 All-to-All、Token 不均衡和跨节点网络拥塞。

## 8. 训练显存优化

### 8.1 Adam 训练状态

以参数量 `Φ` 为单位，混合精度训练通常需要同时保存：

```text
参数：模型计算使用的低精度副本
梯度：反向传播产生的梯度
Master Parameters：优化器更新使用的高精度副本
一阶矩 m：Adam 动量
二阶矩 v：Adam 方差
激活：前向结果
```

因此“参数量 × 参数字节数”只是显存估算的起点，不能直接当作训练显存。

### 8.2 Activation Checkpointing

普通训练保存大量前向激活，反向时直接读取；Checkpointing 只保存部分边界，反向时重新计算中间激活：

```text
普通：Forward 保存所有激活 ──> Backward 直接读取

Checkpoint：Forward 保存少量边界
                    │
                    ▼
             Backward 时重新计算中间结果
```

它用额外计算换显存，适合激活占用明显的深层模型。Checkpoint 粒度、重计算区域和算子随机状态需要保持一致。

### 8.3 ZeRO

ZeRO 通过在数据并行 Rank 之间切分训练状态降低单卡显存：

| 阶段 | 切分内容 | 主要通信 |
|------|----------|----------|
| ZeRO-1 | Optimizer States | 参数更新相关通信 |
| ZeRO-2 | Optimizer States + Gradients | 梯度分片与同步 |
| ZeRO-3 | Parameters + Gradients + Optimizer States | 按需 AllGather/ReduceScatter |

```text
普通 DP：每张卡都有完整参数、梯度和优化器状态
ZeRO-1：只切优化器状态
ZeRO-2：再切梯度
ZeRO-3：参数也按 Rank 切分
```

切分程度越高，显存节省越多，但通信和调度复杂度也会增加。

### 8.4 FSDP

FSDP（Fully Sharded Data Parallel）将参数、梯度和优化器状态分片，并在需要计算某一层时 AllGather 参数，计算完成后释放或重新分片：

```text
参数分片
   │ Forward 前 AllGather 当前层参数
   ▼
执行当前层
   │
   ▼
释放完整参数，保留分片
```

使用 FSDP 时需要关注 Wrap Policy、Prefetch、Mixed Precision、CPU Offload、State Dict 和 Checkpoint 保存格式。

### 8.5 CPU/NVMe Offload

Offload 将部分状态移动到 CPU 内存或 NVMe：

```text
GPU 显存不足
    │
    ├── 参数/优化器状态 → CPU Memory
    └── 更大规模临时状态 → NVMe
```

代价是 PCIe、内存或存储 I/O。Offload 适合显存受限的场景，但需要测量搬运时间是否超过节省的计算时间。

## 9. 推理执行模型

### 9.1 Prefill 与 Decode

自回归 LLM 推理通常分为两个阶段：

```text
输入 Prompt
    │
    ▼
Prefill：并行处理已有 Token，建立 KV Cache
    │
    ▼
Decode：每次生成一个或少量 Token，反复读取 KV Cache
```

| 阶段 | 特点 | 典型瓶颈 |
|------|------|----------|
| Prefill | 序列内并行度高、计算量大 | GPU 算力、输入长度 |
| Decode | 每步 Token 少、需要反复访存 | KV Cache 带宽、调度和延迟 |

常用指标：

- **TTFT**：Time To First Token，从请求到首 Token 的时间。
- **TPOT**：Time Per Output Token，生成阶段每个 Token 的平均耗时。
- **ITL**：Inter-Token Latency，相邻输出 Token 之间的间隔。
- **Throughput**：单位时间生成的 Token 数。

### 9.2 KV Cache

Attention 在生成新 Token 时，历史 Token 的 Key/Value 不需要重复计算，因此缓存下来：

```text
第 t 步：
Q_t 与 K_0...K_t、V_0...V_t 计算 Attention

缓存：K_0...K_t、V_0...V_t
下一步只计算 Q_{t+1}、K_{t+1}、V_{t+1}
```

KV Cache 受以下因素影响：

- 层数和 Head 维度。
- KV Head 数。
- 上下文长度和生成长度。
- 并发请求数。
- KV Cache 数据类型。

### 9.3 PagedAttention

传统 KV Cache 常为每个请求预留一段连续显存，但请求长度不同会产生碎片和预留浪费。PagedAttention 将 KV Cache 切成固定大小的 Block，由逻辑 Block Table 映射到物理 Block：

```text
逻辑序列： [Block 0] [Block 1] [Block 2] [Block 3]
              │         │         │         │
物理显存：   [7]       [2]       [11]      [5]
```

优点：

- 按需分配，减少预留浪费。
- 支持不同请求的 KV Block 共享。
- 请求结束后可以回收 Block。
- 便于调度动态长度请求。

### 9.4 KV Cache 共享

相同前缀的请求可以共享前缀 KV Cache：

```text
请求 A：系统提示词 + 用户问题 A
请求 B：系统提示词 + 用户问题 B
       └── 共享系统提示词对应的 KV Block
```

共享需要处理引用计数、写时复制、Cache Key、失效策略和安全隔离。

## 10. 连续批处理与请求调度

### 10.1 静态批处理

静态批处理等待一组请求凑齐，再一起运行若干步：

```text
请求 A ─────────────────────── 完成
请求 B ─────────── 完成
请求 C ─────────────────────────────── 完成
批次结束后才能开始下一批
```

短请求需要等待长请求，造成设备空闲和尾延迟。

### 10.2 连续批处理

连续批处理按迭代调度请求：

```text
Step 1：A B C
Step 2：A B C + 新请求 D
Step 3：A C D，B 已完成
Step 4：A D + 新请求 E
```

每一步都可以回收完成请求的 KV Block，把资源分配给新请求。

### 10.3 调度器需要管理的资源

- GPU 计算预算。
- KV Cache Block 数量。
- 最大 Batch Token 数。
- 最大并发请求数。
- Prefill 与 Decode 的资源比例。
- 请求优先级、超时和取消。
- P50/P95/P99 延迟目标。

### 10.4 吞吐与延迟的权衡

增加 Batch 往往提高吞吐，但可能增加单请求等待和显存压力：

```text
Batch 太小：GPU 利用率不足，吞吐低
Batch 太大：KV Cache 紧张，排队和尾延迟上升
```

实际服务需要通过负载实验寻找可接受的 Pareto 区间，而不是只追求一个最大吞吐数字。

## 11. 数值精度与量化

### 11.1 混合精度

| 类型 | 特点 | 常见用途 |
|------|------|----------|
| FP32 | 精度和动态范围较好，显存和带宽开销大 | Reference、部分累加 |
| TF32 | 在部分 NVIDIA GPU 上用于加速 FP32 矩阵计算 | 兼顾易用性与吞吐 |
| FP16 | 存储和计算开销低，动态范围较小 | 训练和推理 |
| BF16 | 动态范围接近 FP32，尾数精度低于 FP32 | 训练稳定性较好 |
| FP8 | 更低存储和带宽，需处理缩放 | 大规模训练和推理 |
| INT8 | 整数运算和存储效率高 | 推理量化 |
| INT4 | 压缩率高，误差更敏感 | 大模型权重压缩 |

混合精度并不意味着所有计算都使用低精度：

```text
低精度 Tensor Core 计算
        │
        ├── FP32/BF16 累加或归一化
        ├── Loss Scaling 或动态缩放
        └── Master Weights 使用更高精度更新
```

### 11.2 量化的基本概念

线性量化可以写成：

$$
x_q = \operatorname{round}(x / s) + z
$$

反量化：

$$
x \approx s(x_q - z)
$$

其中 `s` 是 Scale，`z` 是 Zero Point。对称量化通常令 `z=0`；非对称量化可以更好地覆盖偏移后的分布。

### 11.3 量化粒度

| 粒度 | Scale 范围 | 特点 |
|------|------------|------|
| Per-tensor | 整个 Tensor 一个 Scale | 简单，误差可能较大 |
| Per-channel | 每个输出通道一个 Scale | 常用于权重量化 |
| Per-group | 每组权重一个 Scale | 精度和开销折中 |
| Token-wise | 每个 Token 动态计算 | 适合激活但有运行时开销 |

### 11.4 PTQ 与 QAT

- **PTQ（Post-Training Quantization）**：训练完成后使用校准数据估计 Scale 并量化。
- **QAT（Quantization-Aware Training）**：训练过程中模拟量化误差，让模型适应低精度。

```text
PTQ：训练模型 → 校准 → 量化 → 评估
QAT：插入量化模拟 → 继续训练 → 导出量化模型
```

### 11.5 GPTQ、AWQ、SmoothQuant 与 FP8

| 方法 | 核心思路 | 主要对象 |
|------|----------|----------|
| GPTQ | 使用近似二阶信息逐层校正权重误差 | 权重，常见低比特量化 |
| AWQ | 识别更重要的权重通道并保护其精度 | 权重 |
| SmoothQuant | 将激活中的异常值平滑到权重侧 | 激活 + 权重 |
| FP8 | 使用低位浮点表示并管理动态 Scale | 训练/推理计算 |

选择量化方法时需要同时评估：

- 目标硬件是否有对应低精度指令。
- 推理框架是否支持该格式。
- 权重、激活和 KV Cache 是否分别量化。
- 校准集是否覆盖真实输入分布。
- 精度、吞吐、显存和延迟是否满足约束。

## 12. 推理服务架构

### 12.1 基本组件

```text
Client
  │ HTTP/gRPC
  ▼
API Gateway / Load Balancer
  │ 鉴权、限流、路由
  ▼
Request Scheduler
  │ Continuous Batching、优先级、取消
  ▼
Model Runtime
  │ Prefill、Decode、KV Cache、量化 Kernel
  ▼
GPU Worker
  │
  ├── 模型权重
  ├── KV Cache
  └── CUDA/NCCL/Tensor Parallel
```

### 12.2 服务指标

| 指标 | 含义 |
|------|------|
| QPS/RPS | 每秒请求数 |
| Input Tokens/s | 输入 Token 处理速度 |
| Output Tokens/s | 输出 Token 生成速度 |
| TTFT | 首 Token 延迟 |
| TPOT/ITL | 输出 Token 间隔 |
| E2E Latency | 端到端请求延迟 |
| Queue Time | 调度前排队时间 |
| GPU Utilization | GPU 利用率 |
| KV Cache Usage | KV Block 使用比例 |
| OOM Rate | 显存不足请求比例 |
| Error Rate | 请求错误比例 |

### 12.3 吞吐测试

测试必须固定并记录：

```text
模型版本、Tokenizer 版本
GPU 型号、数量和并行配置
输入长度分布、输出长度分布
并发数和请求到达模式
Batch Token 上限和 KV Cache 配置
Warmup 次数、测试时长和采样窗口
```

只测试一个固定短 Prompt 得到的结果，不能代表真实长上下文和混合长度负载。

### 12.4 服务扩缩容

扩容触发条件可以组合：

- 请求队列长度。
- GPU 利用率。
- P95/P99 延迟。
- KV Cache 使用率。
- 错误率和 OOM 率。

缩容时要考虑模型加载时间、Checkpoint/权重缓存、正在执行的请求和连接排空。

## 13. 监控、Profiling 与故障定位

### 13.1 监控分层

```text
业务层：请求数、错误率、Token 吞吐、延迟
服务层：队列、Batch、KV Cache、Worker 状态
框架层：算子耗时、显存分配、通信等待
设备层：GPU 利用率、显存、温度、功耗、ECC
系统层：CPU、内存、磁盘、网络、NUMA
```

### 13.2 GPU 设备观察

```bash
nvidia-smi
nvidia-smi dmon
nvidia-smi pmon -s um
nvidia-smi topo -m
```

### 13.3 时间线分析

Nsight Systems 适合回答：

- CPU 是否在等待 GPU？
- H2D、Kernel、D2H 是否重叠？
- 多个 Stream 是否产生并发？
- 分布式通信是否阻塞了计算？

```bash
nsys profile -o timeline python train.py
nsys stats timeline.nsys-rep
```

Nsight Compute 适合回答单个 Kernel 的问题：

```bash
ncu --set full -o kernel_report ./program
```

重点指标包括 Occupancy、Global Memory Efficiency、DRAM Throughput、Cache 命中、Warp Stall、Shared Memory Conflict 和指令吞吐。

### 13.4 常见故障定位路径

#### GPU 不可见

```text
nvidia-smi
  ├── 失败：先处理驱动、设备或宿主机问题
  └── 成功
       ├── 容器能否访问 /dev/nvidia*
       ├── CUDA_VISIBLE_DEVICES 是否正确
       ├── 框架是否检测到 CUDA
       └── 用户是否有设备权限
```

#### 显存不足

```text
确认显存使用者
  ├── 权重过大
  ├── KV Cache 过大
  ├── Batch/序列长度过大
  ├── 激活或临时 Buffer 过大
  ├── 碎片化
  └── 其他进程占用
```

对应措施包括降低 Batch、限制最大 Token、使用量化、启用 Checkpointing、分片参数、调整 KV Cache Block 和清理无效进程。

#### 分布式训练卡住

依次检查：

1. 所有 Rank 是否都启动并进入同一个 Collective。
2. `MASTER_ADDR`、`MASTER_PORT`、Rank 和 World Size 是否一致。
3. 防火墙和端口是否允许通信。
4. NCCL 是否选择了正确网卡和 GPU 拓扑。
5. 是否存在某个 Rank 数据加载或 OOM 后提前退出。
6. 是否有 CPU/NUMA/网卡中断造成极端延迟。

### 13.5 可复现性

记录以下信息：

- Git Commit、模型和数据版本。
- 容器镜像摘要。
- 依赖锁文件。
- GPU、驱动、Toolkit、通信库版本。
- 随机种子和确定性开关。
- 并行拓扑、Batch、学习率和精度配置。

## 14. 学习实验路线

### 14.1 单机性能实验

1. 使用 CPU Reference 验证向量加法和矩阵乘。
2. 使用 CUDA Event 测量 Kernel 时间。
3. 改变矩阵尺寸和数据类型。
4. 使用 Nsight Compute 观察带宽和 Occupancy。
5. 对比手写 Kernel 与 cuBLAS。

### 14.2 分布式通信实验

1. 运行单进程单卡基线。
2. 运行单机多卡 AllReduce。
3. 记录消息大小、带宽和延迟。
4. 改变 Rank 数，观察扩展效率。
5. 对比不同 GPU 拓扑和网络接口。

### 14.3 推理服务实验

1. 固定模型和 Tokenizer，建立单请求基线。
2. 测量 Prefill 与 Decode 的时间。
3. 改变输入长度、输出长度和并发数。
4. 观察 KV Cache 使用率和 P95 延迟。
5. 对比静态批处理和连续批处理。
6. 对比 FP16/BF16 与量化模型的精度、显存和吞吐。

## 15. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FlashAttention Repository](https://github.com/Dao-AILab/flash-attention)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
