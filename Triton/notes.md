# Triton GPU Kernel 编程

> Triton 是面向 GPU Kernel 的 Python DSL 与编译器。它通过“以块为单位的程序实例（program instance）”表达并行计算，让开发者能够在 PyTorch 张量之上编写、融合和调优自定义算子。

Triton 的价值不在于替代某一个框架，而在于填补通用框架算子与手写 CUDA Kernel 之间的实现层：当一个计算由多个小算子组成、存在中间张量读写，或需要针对输入形状调整并行策略时，可以用一个 Triton Kernel 表达该计算。常见学习对象包括向量运算、Softmax、LayerNorm、矩阵乘与 Attention 的局部计算。

官方稳定二进制发行版以 Linux 为支持平台；当前主仓库列出的 NVIDIA 目标为 Compute Capability 8.0+，AMD 目标为 ROCm 6.2+。在 Windows 上整理和阅读本笔记不要求安装 Triton；实际执行 Kernel 前，应在 Linux/WSL 或经验证的目标环境中核对当前版本的兼容性、PyTorch 与驱动组合。[官方兼容性](https://github.com/triton-lang/triton#compatibility)

## 内容概览

- Triton、CUDA、PyTorch、TensorRT 与 vLLM 的分层关系
- Blocked Programming、Program Instance、Grid、`tl.load` 与 `tl.store`
- `@triton.jit`、运行时参数、编译期元参数与 Kernel 启动方式
- 向量加法、二维索引、归约、矩阵乘与算子融合的基本模式
- 掩码、访存、Warp、流水线、自动调优与进阶 Layout
- 正确性测试、基准测试、解释器与 GPU 调试流程
- 适合作为后续小型 Demo 的学习路线

## 1. Triton 在 GPU 软件栈中的位置

```text
Python 应用 / PyTorch Module / 推理服务
                 │
     PyTorch eager、torch.compile、vLLM 等调用方
                 │
        Triton Kernel / CUDA Kernel / 库算子
                 │
    CUDA 或 ROCm Runtime、驱动、GPU 硬件
```

| 组件 | 主要抽象 | 典型产物 | 学习时关注点 |
|---|---|---|---|
| CUDA C++ | 线程、Block、Grid、显式内存与同步 | `.cu` Kernel | 硬件执行模型与底层控制 |
| Triton | 分块张量程序与 Program Instance | Python 中的 JIT Kernel | 算子融合、数据布局、形状专用化 |
| PyTorch | Tensor 与算子图 | 模型/计算图 | 正确性基线、张量生命周期、集成入口 |
| TensorRT | 模型图优化与 Engine 执行 | TensorRT Engine | 部署图、精度、Profile 与运行时 |
| vLLM | LLM 调度与服务运行时 | HTTP 服务与批处理执行 | 请求调度、KV Cache、吞吐与延迟 |

一个 Triton Kernel 只是计算链路中的一个 GPU 执行单元：它可以被 PyTorch 或推理框架调用，也可以融合若干逐元素/归约步骤；Engine 构建、请求调度、KV Cache 管理仍属于更上层的部署和服务问题。

## 2. 先建立术语表

| 术语 | 含义 |
|---|---|
| Kernel | 在 GPU 上并行执行的函数；Triton Kernel 使用 `@triton.jit` 标记。 |
| Program Instance | 一次 Kernel 启动中的一个分块程序实例，可类比 CUDA 的一个线程块，但不要求开发者逐线程编程。 |
| Grid | 全部 Program Instance 的逻辑网格；一维 Grid 常写为 `(program_count,)`。 |
| `tl.program_id(axis)` | 当前 Program Instance 在某个 Grid 轴上的编号。 |
| Block | 一个 Program Instance 一次处理的一块元素，例如长度为 `BLOCK_SIZE` 的向量片段。 |
| Pointer Tensor | 由基地址与偏移组成的一组地址，可被 `tl.load`/`tl.store` 批量访问。 |
| Mask | 与 Block 同形状的布尔条件；用于处理不能填满完整 Block 的边界元素。 |
| Meta-parameter | 编译期已知的参数，例如 `BLOCK_SIZE: tl.constexpr`；改变它通常会生成新的 Kernel 专用版本。 |
| `num_warps` / `num_stages` | Kernel 配置项，影响并行执行和流水线策略；需要通过测量决定。 |
| Layout | 进阶的 tile 元素映射概念；在实验性 Gluon 模型中显式描述 CTA、Warp、lane 和寄存器的分配。 |

## 3. 编程模型：先描述块，再映射到硬件

CUDA 常从单线程索引出发，再把线程组织成 Block；Triton 的核心抽象是直接为一个 Block 写标量/向量化的程序。开发者需要明确每个 Program Instance 处理哪一块数据、块内数据如何访问、哪些元素是边界，而线程到硬件资源的具体映射主要由编译器完成。

```text
长度 N 的向量加法，BLOCK_SIZE = 256

grid 中的 pid:     0             1             2        ...
负责的元素:     [0, 255]     [256, 511]    [512, 767]   ...
offsets:       pid * 256 + tl.arange(0, 256)
```

对一维数据，常见的四步是：

1. 用 `tl.program_id(0)` 取得当前块编号。
2. 用 `tl.arange(0, BLOCK_SIZE)` 生成块内偏移。
3. 用基地址加偏移形成 Pointer Tensor，再用 `tl.load` 读取。
4. 计算后用 `tl.store` 写回；最后一块不足时用 `mask` 保护访问。

Triton 的编程模型和编译器会围绕这种分块结构分析数据流、局部性和并行度。它并不免除对访存、寄存器、并行粒度和实际形状的分析；分块大小是否合适仍要由正确性和基准测试共同验证。[编程模型介绍](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)

## 4. Kernel 的定义与启动

### 4.1 `@triton.jit` 与参数分类

```python
import triton
import triton.language as tl


@triton.jit
def example_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # values 是一个 Block，不是 Python 标量。
```

| 参数类别 | 示例 | 何时确定 | 用途 |
|---|---|---|---|
| 指针参数 | `x_ptr` | 启动时 | 指向 GPU 上的输入或输出张量存储。 |
| 运行时标量 | `n_elements`、`stride` | 启动时 | 参与地址计算、循环边界或条件判断。 |
| 编译期元参数 | `BLOCK_SIZE: tl.constexpr` | 编译专用版本时 | 决定 Block 形状、静态分支和展开机会。 |
| 启动配置 | `num_warps`、`num_stages` | 启动/编译时 | 提供并行与流水线配置。 |

不要把 `tl.constexpr` 理解为普通 Python 常量：它是影响 Kernel 专用化的编译期信息。不同的 `BLOCK_SIZE`、数据类型、布局或部分配置都可能触发额外编译；因此基准测试需把首次编译时间与稳态运行时间分开记录。

### 4.2 最小向量加法

下面的示例构成后续第一个 Demo 的核心 Kernel。代码假设 `x` 和 `y` 是相同形状、连续且位于同一 GPU 的一维 PyTorch Tensor。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape or x.device != y.device or not x.is_cuda:
        raise ValueError("x 和 y 必须是同形状、同设备的 CUDA Tensor")

    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
    return output


# 未来 Demo 的正确性基线：覆盖不能整除 BLOCK_SIZE 的长度。
x = torch.randn(1000, device="cuda")
y = torch.randn(1000, device="cuda")
torch.testing.assert_close(triton_add(x, y), x + y)
```

`add_kernel[grid](...)` 是 Kernel 启动语法。这里 `grid` 依赖 `BLOCK_SIZE`，因此通过 `meta` 计算需要多少个 Program Instance。`triton.cdiv(a, b)` 表示向上整除；`mask` 确保最后一个 Program Instance 不会访问或写入超过 `n_elements` 的地址。官方的入门教程以该模式逐步引入 Block、Mask、测试与性能比较。[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

## 5. 访问内存：Pointer、Block 与边界

### 5.1 `tl.load` 和 `tl.store`

```python
@triton.jit
def scale_kernel(x_ptr, y_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, alpha * x, mask=mask)
```

- `x_ptr + offsets` 产生一组地址，而不是单个地址。
- `tl.load` 从这些地址形成一个 Block；`mask=False` 的位置不访问内存，并以 `other` 作为结果。
- `tl.store` 只对 `mask=True` 的地址写入。
- 边界 Mask 解决“元素数量不是 `BLOCK_SIZE` 整数倍”的问题，不代替形状、stride、dtype 与别名关系的检查。

### 5.2 二维 Tensor 的索引骨架

二维计算的关键是把逻辑行列坐标转换为线性地址。对连续行主序矩阵 `X[M, N]`，第 `row, col` 个元素地址可写作 `X + row * stride_m + col * stride_n`。

```python
@triton.jit
def row_scale_kernel(
    x_ptr, y_ptr, m_size, n_size, stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = (row < m_size) & (cols < n_size)
    offsets = row * stride_m + cols * stride_n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, 0.5 * x, mask=mask)
```

调用方应传入实际 Tensor 的 stride，而不是假设所有 Tensor 连续。若决定只支持连续输入，应在 Python 包装层显式检查或转换，并将这部分代价计入端到端测量。

### 5.3 版本演进：Block Pointer 与 Layout

阅读旧示例时可能会遇到 `tl.make_block_ptr`、`tl.advance` 与 `boundary_check`。它们用于从父 Tensor 的形状、stride 和块起点构造 Block Pointer；和普通 Pointer Tensor 不同，Block Pointer 的越界策略由 `boundary_check` 表达，不能同时传入普通 `mask`。但官方 v3.7.0 release notes 已说明 `make_block_ptr` 会产生弃用警告，因此第一阶段 Demo 不以它作为新代码入口，优先使用本笔记中的 `tl.arange`、普通 Pointer Tensor 和 Mask 模式。[v3.7.0 release notes](https://github.com/triton-lang/triton/releases/tag/v3.7.0)

另一个容易混淆的概念是显式 Layout。它属于实验性 Gluon 的更低层编程模型，描述 tile 元素怎样分配到 CTA、Warp、lane 与寄存器，和“从父 Tensor 取一个逻辑块”的 Block Pointer 不是同一层抽象。学习基础 Kernel 时先观察 Block 大小、访问模式和 `num_warps` 对性能的影响；后续需要研究显式数据布局、线程间通信或共享内存 bank conflict 时，再单独进入 Gluon Layout。[Gluon Layout 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)

## 6. 从逐元素算子到融合算子

### 6.1 为什么融合可能更快

以 `y = relu(x + bias)` 为例，若拆成两个独立 GPU 算子，需要先把 `x + bias` 写入中间张量，再读出并执行 `relu`。单个 Triton Kernel 可在寄存器中的同一 Block 内完成加法与激活，只写一次最终输出。

```python
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, y_ptr, n_elements, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cols = offsets % n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, tl.maximum(x + bias, 0.0), mask=mask)
```

融合减少中间张量的全局内存读写和 Kernel launch，但并不保证一定更快：额外寄存器、复杂控制流、低复用数据或成熟库算子的高度优化都可能改变结果。评估时应同时报告基线、张量形状、dtype、预热策略和端到端耗时。

### 6.2 归约与 Softmax 的结构

归约会把一组元素合成一个或少量结果，例如 `sum`、`max`、`mean`。Softmax 常按行执行：

```text
m = max(x_i)
z_i = exp(x_i - m)          # 减去 m，避免 exp 溢出
y_i = z_i / sum(z_i)
```

当一行可以放入一个合理大小的 Block 时，一个 Program Instance 可加载整行、完成 `tl.max` / `tl.sum` 归约并写回结果。行宽过大时，需要拆分、分阶段归约或采用不同算法；Block 并非越大越好，因为寄存器、占用率和访存模式都会变化。[融合 Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

### 6.3 矩阵乘的结构

矩阵乘将输出矩阵切成 `BLOCK_M × BLOCK_N` 的 tile；每个 Program Instance 沿 K 维按 `BLOCK_K` 迭代，累加局部结果。

```text
C[M, N] 的一个 tile
    accumulator[BLOCK_M, BLOCK_N] = 0
    for k in range(0, K, BLOCK_K):
        A_tile = A[rows, k:k+BLOCK_K]
        B_tile = B[k:k+BLOCK_K, cols]
        accumulator += tl.dot(A_tile, B_tile)
    store accumulator -> C[rows, cols]
```

矩阵乘是学习 `tl.dot`、二维指针算术、L2 局部性、`GROUP_SIZE_M`、`num_warps` 与自动调优的合适对象，但不适合作为第一份可运行 Demo。先建立向量加法和 Softmax 的正确性/基准流程，再分析官方矩阵乘教程中的 tile 配置与重排策略。[矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

## 7. 性能模型与调优入口

### 7.1 优先判断瓶颈类型

| 现象 | 常见瓶颈 | 首先检查 |
|---|---|---|
| 简单逐元素算子带宽接近上限前增长缓慢 | Global Memory 带宽 | 合并访问、读写次数、融合、中间张量。 |
| 小 Tensor 延迟占主导 | Kernel launch 与调度开销 | 是否需要融合、批处理或图执行。 |
| 大量乘加、`tl.dot` 占主导 | 计算吞吐 | Tile 尺寸、Warp 数、数据类型、Tensor Core 路径。 |
| 增大 Block 后反而变慢 | 寄存器/共享资源压力 | occupancy、寄存器用量、访存和实际吞吐。 |
| 输入形状变化时性能波动 | 专用化/调优配置不匹配 | 基准形状覆盖、`@triton.autotune` 的 key。 |

Roofline 思路仍然适用：估算该算子的总 FLOPs、读写字节数与算术强度（FLOPs/byte），再用测量结果判断更接近计算上限还是带宽上限。不要只看 occupancy；它反映可活跃的并行度，并不能单独代表性能。

### 7.2 自动调优

`@triton.autotune` 会为指定的输入键测试多个 `triton.Config`，选择较快配置并缓存结果。它适合形状或 dtype 会显著改变最佳 Block/Warp 选择的 Kernel。

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def tuned_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

自动调优会多次运行候选 Kernel。若 Kernel 累加或原地修改输出，必须设计可重复的初始化，或使用 `reset_to_zero` / 恢复逻辑，避免把调优本身写成错误结果。`key` 应包含会影响最佳配置的输入维度或属性，不能把所有参数机械加入。[`triton.autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)

## 8. 正确性与性能测量

### 8.1 正确性先行

每个 Kernel 至少覆盖以下维度：

| 检查项 | 示例 |
|---|---|
| 参考实现 | `torch` 原生算子或清晰的 Python/PyTorch 组合。 |
| 形状 | 极小、典型、很大、不能整除 Block 的长度、二维非方阵。 |
| dtype | `float32` 作为初始基线，再按需要加入 `float16` / `bfloat16`。 |
| 数值阈值 | 用 `torch.testing.assert_close` 指定 `atol` / `rtol`。 |
| 输入布局 | 连续与非连续输入；若不支持，验证包装层能明确拒绝或转换。 |
| 边界 | 空输入策略、尾部 Mask、stride、NaN/Inf 的预期语义。 |

浮点归约的加法顺序与 PyTorch 参考实现可能不同，低精度下误差也会累积。因此结果不应使用无条件精确相等判断；阈值必须结合 dtype、归约长度和算法语义设置。

### 8.2 基准测试的最小框架

```python
import triton.testing


def gb_per_second(n_elements: int, element_size: int, ms: float) -> float:
    # 向量加法：读 x、读 y、写 output，共 3 次张量读写。
    transferred_bytes = 3 * n_elements * element_size
    return transferred_bytes * 1e-9 / (ms * 1e-3)


ms = triton.testing.do_bench(lambda: triton_add(x, y), quantiles=[0.2, 0.5, 0.8])
low_ms, median_ms, high_ms = ms
print("p20/p50/p80 (ms):", low_ms, median_ms, high_ms)
print("p50 GB/s:", gb_per_second(x.numel(), x.element_size(), median_ms))
```

`triton.testing.do_bench` 用于测量一个可调用对象的运行时间，默认返回均值毫秒数；传入 `quantiles` 时可得到时间分位数。对比时至少保持输入、dtype、设备、预热、是否计入分配、是否同步和输出校验方式一致。首次调用可能包含 JIT 编译，不能与稳态耗时混在一起。[`do_bench` API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)

## 9. 调试路径

### 9.1 分层定位

```text
导入/安装失败
  -> 核对 OS、Python、驱动、PyTorch、Triton 版本与目标后端

Kernel 编译失败
  -> 缩小为最小输入；检查 Python 包装层、参数类型、constexpr 与 Triton API

结果错误
  -> 与 PyTorch 参考比较；覆盖尾部 Mask、stride、dtype；开启解释器或设备断言

结果正确但较慢
  -> 先排除首次编译、分配和同步；再测形状、带宽/FLOPs 与 Profile
```

### 9.2 Triton 解释器与断言

官方解释器可让 Triton Kernel 绕过 GPU 编译，在 CPU 上用 NumPy 等价操作顺序模拟 Program Instance，适合检查索引和中间值。PowerShell 中可临时设置：

```powershell
$env:TRITON_INTERPRET = "1"
python .\your_kernel_test.py
Remove-Item Env:TRITON_INTERPRET
```

解释器用于逻辑调试，不能用于性能结论；它对 `bfloat16` 和间接指针访问等模式存在限制。编译期问题可使用 `tl.static_print`、`tl.static_assert`，设备运行期问题可使用 `tl.device_print`、`tl.device_assert`；其中 `tl.device_assert` 需要设置 `TRITON_DEBUG=1` 才执行。NVIDIA GPU 上还可使用 `compute-sanitizer` 排查越界访问和数据竞争。[官方调试指南](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)

## 10. 安装与环境核对

### 10.1 核对，而非直接修改当前学习环境

Triton 会与 PyTorch、驱动和目标 GPU 后端共同决定可运行性。为避免把 GPU 编译链依赖混入当前笔记环境，建议在后续实际 Demo 前单独创建项目级环境，并记录下列信息：

```powershell
# 目的：记录 Python、PyTorch 与 CUDA 可见性；预期：输出版本、GPU 名称与 capability。
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')"

# 目的：确认 NVIDIA 驱动是否看到目标 GPU；预期：GPU、驱动版本、显存和进程信息。
nvidia-smi
```

判断标准：只有 Python 能导入目标 PyTorch、`torch.cuda.is_available()` 为 `True`、目标 GPU/后端满足当期 Triton 兼容性，并且实际环境的安装策略已经确认后，才进入安装与运行步骤。官方 Python 包的安装命令为 `pip install triton`；具体 wheel、后端、Python 版本与平台边界应以当前官方安装页和兼容性说明为准。[官方安装页](https://triton-lang.org/main/getting-started/installation.html)

### 10.2 缓存与可复现性

- 首次运行可能下载/生成编译相关产物并建立缓存；记录 Triton、PyTorch、驱动、GPU 架构和输入形状。
- 升级 Triton、PyTorch、驱动或更换 GPU 后，应重新运行正确性测试与稳态基准。
- 不把某台机器上首次编译成功或某个单一形状的最快配置推广为通用结论。
- Kernel 的可复现记录至少应包含：源码版本、dtype、形状/stride、`BLOCK_SIZE`、`num_warps`、`num_stages`、测试阈值与测量方法。

## 11. 后续 Demo 路线

本轮仅沉淀笔记，不创建或运行 GPU Demo。后续可按以下顺序建立彼此独立、可验证的最小实验：

| 阶段 | Demo | 核心概念 | 验收输出 |
|---|---|---|---|
| 1 | `vector_add` | `@triton.jit`、Grid、Block、Mask、`tl.load/store` | 与 `torch.add` 一致；长度覆盖 `1`、`1000`、大向量。 |
| 2 | `fused_bias_relu` | 融合、广播列索引、端到端带宽 | 与 PyTorch 组合算子一致；报告稳态 p50 与 GB/s。 |
| 3 | `row_softmax` | 行级归约、数值稳定性、Block 限制 | 与 `torch.softmax` 在设定容差内一致；覆盖多种列宽。 |
| 4 | `tiled_matmul` | 二维 tile、`tl.dot`、自动调优 | 与 `torch.matmul` 一致；对比形状与 TFLOPS。 |

每个 Demo 固定使用同一交付结构：`reference -> kernel -> correctness tests -> warmup -> steady-state benchmark -> result record`。这样能够把“能运行”“结果一致”“为何快/慢”三个问题分开验证，并为后续 CUDA、TensorRT 与 vLLM 的学习保留可比较的性能记录。

## 12. 推荐阅读顺序

1. 本文第 1～5 节：理解 Block、Grid、Pointer 与 Mask。
2. 官方 Vector Addition：完成第一个正确性对照。
3. 官方 Fused Softmax：学习归约和数值稳定性。
4. 官方 Matrix Multiplication：学习 tile、数据局部性与自动调优。
5. 官方 LayerNorm / Fused Attention：把融合、归约和训练/推理热点联系起来。
6. 将自定义 Kernel 放回 PyTorch 调用路径，区分 Kernel 时间、算子时间与端到端时间。

## 官方资料

- [Triton 主仓库与兼容性](https://github.com/triton-lang/triton)
- [Installation](https://triton-lang.org/main/getting-started/installation.html)
- [Programming Guide: Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Gluon Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)
- [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html) 与 [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
