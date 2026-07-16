# CUDA 并行计算

> NVIDIA CUDA 编程学习笔记

CUDA 是 NVIDIA 提供的通用并行计算平台和编程模型。它将 CPU 作为 Host，将 GPU 作为 Device，通过 Kernel、线程层级、显存层级和异步执行机制，把适合并行化的计算映射到 GPU 上。

这份笔记以 CUDA C++ 为主，穿插 PyTorch、Numba 和 CUDA 工具链示例。代码重点用于说明执行过程；真正运行时需要根据 GPU 型号、CUDA Toolkit、编译器和驱动版本调整命令。

## 先建立 CUDA 术语表

- **Host / Device**：Host 通常指 CPU 及其内存，Device 通常指 GPU 及其显存。一个 CUDA 程序会在两侧之间准备、复制和处理数据。
- **Kernel**：由 Host 发起、在 Device 上由大量线程执行的函数，通常用 `__global__` 修饰。
- **Thread / Block / Grid**：Thread 是单个执行实例；Block 是可以在同一个 SM 上协作的一组线程；Grid 是一次 Kernel 启动创建的全部 Block。
- **SM（Streaming Multiprocessor）**：GPU 中负责调度线程块、执行指令并管理片上资源的硬件单元。
- **Warp**：SM 调度线程的常见硬件分组，通常包含 32 个线程。一个 Block 会被拆成多个 Warp。
- **SIMT**：Single Instruction, Multiple Threads，即程序以线程为单位表达逻辑，硬件按 Warp 组织执行。
- **Global / Shared / Register Memory**：Global Memory 容量大但延迟较高；Shared Memory 由同一 Block 共享；Register 是单线程私有的高速存储。
- **Stream**：按顺序排列的一组 GPU 操作；不同 Stream 中无依赖的操作可能并发。
- **Occupancy**：一个 SM 上活跃 Warp 数量相对于硬件上限的比例，反映资源允许的并发程度，但不是单独的性能指标。
- **Memory-Bound / Compute-Bound**：前者主要受数据搬运速度限制，后者主要受算术吞吐限制。

后文中的每个 CUDA 代码块都可以按“数据在哪一侧 → 哪些线程负责哪些元素 → 是否需要协作/同步 → 结果如何回到 Host”的顺序阅读。

## 内容概览

- CUDA 平台、Host/Device 与程序执行链
- CUDA Toolkit、`nvcc`、Driver 和计算能力
- Kernel、Thread、Block、Grid、Warp 与 SIMT
- Global、Shared、Register、Constant、Local 等内存
- 同步、原子操作、Stream、Event 与异步执行
- 向量运算、归约、矩阵乘、转置等并行模式
- 访存合并、Bank Conflict、Occupancy 与性能测量
- CUDA 错误检查、`compute-sanitizer` 与数值正确性
- CUDA 库、PyTorch CUDA、Numba CUDA 与自定义扩展

## 1. CUDA 程序的基本结构

### 1.1 异构计算模型

CUDA 程序运行在 CPU 和 GPU 组成的异构系统中：

```text
Host（CPU）                          Device（GPU）
┌────────────────────┐              ┌────────────────────┐
│ 主机代码            │              │ Kernel              │
│ Host Memory        │ -- H2D -->  │ Device Memory      │
│ CUDA Runtime API   │ <-- D2H --  │ SM / Warp / Thread  │
└────────────────────┘              └────────────────────┘
```

典型执行链如下：

```text
1. CPU 准备输入数据
2. CPU 在 Host Memory 中分配或加载数据
3. 通过 CUDA API 将数据复制到 Device Memory
4. CPU 启动 GPU Kernel
5. GPU 上的线程并行执行 Kernel
6. CPU 等待或查询 GPU 工作是否完成
7. 将结果复制回 Host Memory
8. CPU 检查结果并释放资源
```

GPU 不会自动接管整个 C++ 程序。只有标记为 Kernel 的函数和显式调用的设备函数会在 GPU 上执行，其他普通 C++ 代码仍然运行在 CPU 上。

### 1.2 CUDA 生态中的主要组件

| 组件 | 作用 |
|------|------|
| NVIDIA Driver | 让操作系统和应用能够访问 GPU |
| CUDA Toolkit | 开发工具、头文件、库、编译器和调试工具的集合 |
| CUDA Runtime API | 负责设备管理、内存、Kernel、Stream、Event 等运行时操作 |
| CUDA Driver API | 更底层的模块加载、Context 和 Kernel 管理接口 |
| `nvcc` | CUDA 编译器驱动，协调 Host 编译器和 Device 编译流程 |
| PTX | 面向虚拟 GPU ISA 的中间表示 |
| SASS | 面向具体 GPU 架构的机器指令 |
| cuBLAS | GPU 上的 BLAS 线性代数库 |
| cuDNN | 深度学习常用算子库 |
| cuFFT | GPU 快速傅里叶变换库 |
| Thrust | 类似 STL 的 CUDA C++ 模板库 |
| CUB | 面向 CUDA 的底层并行原语库 |
| Nsight Systems | 分析 CPU、GPU、Stream 和整体时间线 |
| Nsight Compute | 分析单个 Kernel 的指令、访存、Occupancy 和瓶颈 |
| Compute Sanitizer | 检查越界访问、竞争条件、初始化和同步问题 |

### 1.3 检查 CUDA 环境

#### 使用 `nvidia-smi` 查看驱动和设备

```bash
nvidia-smi
```

常见字段包括 GPU 型号、Driver Version、CUDA Version、显存总量、已用显存和进程列表。`CUDA Version` 表示驱动声明支持的 CUDA Runtime 上限，不一定等于本机 Toolkit 版本。

#### 使用 `nvcc` 查看 Toolkit 编译器

```bash
nvcc --version
where nvcc       # Windows
which nvcc       # Linux
```

`nvidia-smi` 和 `nvcc --version` 展示的是不同层次的信息，不能简单用其中一个代替另一个。

#### 使用 Python 检查 PyTorch CUDA

```python
import torch

# is_available 检查当前 Python 进程是否能访问 CUDA 设备。
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("built CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    # 只有设备可用时，才能查询数量和设备名称。
    print("device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))
```

`torch.version.cuda` 表示 PyTorch 构建时使用的 CUDA 版本；它与系统安装的 Toolkit 版本可能不同。编译 PyTorch 扩展时，还需要关注本机 `nvcc` 与 PyTorch 二进制的兼容性。

### 1.4 CUDA 文件与编译命令

CUDA C++ 源文件通常使用 `.cu` 后缀：

```text
main.cpp       普通 C++ Host 代码
kernel.cu      可以包含 Host 和 Device 代码的 CUDA 源文件
header.cuh     常用于 CUDA 相关的 C++ 头文件
```

最小编译命令：

```bash
nvcc vector_add.cu -o vector_add
```

指定 C++ 标准并保留调试信息：

```bash
nvcc -std=c++17 -g -G vector_add.cu -o vector_add_debug
```

常用编译选项：

| 选项 | 作用 |
|------|------|
| `-std=c++17` | 指定 Host 端 C++ 标准 |
| `-O3` | 开启 Host 端优化 |
| `-G` | 生成较适合 Device 调试的代码，通常会降低性能 |
| `-lineinfo` | 保留源代码行号，便于性能分析 |
| `-arch=sm_80` | 为指定的真实 GPU 架构生成代码 |
| `-gencode` | 同时指定多个虚拟架构和真实架构 |
| `-Xcompiler` | 向 Host 编译器传递参数 |
| `-Xptxas=-v` | 输出寄存器、Shared Memory 等资源使用情况 |

```bash
nvcc -O3 -lineinfo -Xptxas=-v vector_add.cu -o vector_add
```

### 1.5 编译流程：Host 代码、PTX 与 SASS

`nvcc` 是协调 Host 和 Device 编译过程的驱动程序：

```text
             .cu
              │
              ▼
          nvcc 分离代码
        ┌───────────────┐
        │ Host C++      │ ── Host 编译器 ── Host 目标文件
        │ Device CUDA   │ ── PTXAS       ── SASS / Fatbin
        └───────────────┘
              │
              ▼
           链接生成可执行文件
```

相关概念：

- **PTX**：虚拟指令集，具有一定的前向兼容能力，可以在运行时进一步编译。
- **SASS**：针对具体 GPU 架构生成的真实指令。
- **Compute Capability**：GPU 的计算能力版本，通常写成 `major.minor`，编译参数里常见 `sm_XX`。
- **Fat Binary**：将多个架构的 Device 代码打包到一个二进制中。

```bash
nvcc -ptx vector_add.cu -o vector_add.ptx
nvcc -Xptxas=-v vector_add.cu -o vector_add
```

## 2. Kernel 与线程层级

### 2.1 Kernel 是什么

Kernel 是由 CPU 发起、在 GPU 上由大量线程执行的函数，使用 `__global__` 修饰：

```cpp
// ===== 最小 Kernel：从 Host 启动一个 GPU 线程 =====
__global__ void hello_kernel() {
    // 这段函数体在 Device 上执行；每个线程都会执行一次。
    printf("hello from GPU\\n");
}

int main() {
    // <<<grid, block>>> 创建 1 个 Block，Block 中只有 1 个 Thread。
    hello_kernel<<<1, 1>>>();
    // Kernel 启动通常异步，显式同步后再退出才能看到完整执行结果。
    cudaDeviceSynchronize();
    return 0;
}
```

CUDA 函数修饰符：

| 修饰符 | 调用位置 | 执行位置 |
|--------|----------|----------|
| `__global__` | Host | Device |
| `__device__` | Device | Device |
| `__host__` | Host | Host |
| `__host__ __device__` | Host 或 Device | Host 或 Device |

Kernel 通常返回 `void`。Kernel 启动本身是异步的，后续 Host 代码可能在 GPU 完成前继续执行。

### 2.2 Thread、Block 与 Grid

```text
Grid：一次 Kernel 启动创建的全部线程
└── Block：可以在同一个 SM 上协作的线程集合
    └── Thread：执行 Kernel 的基本单位
```

| 层级 | 内建变量 | 含义 |
|------|----------|------|
| Thread | `threadIdx` | 当前线程在 Block 内的坐标 |
| Block | `blockIdx` | 当前 Block 在 Grid 内的坐标 |
| Block 尺寸 | `blockDim` | 一个 Block 在各维度上的线程数 |
| Grid 尺寸 | `gridDim` | 一个 Grid 在各维度上的 Block 数 |

```cpp
kernel<<<grid, block>>>(arguments);
kernel<<<grid, block, dynamic_shared_bytes, stream>>>(arguments);
```

### 2.3 一维全局索引

一维问题中，一个线程负责一个元素时，常用全局索引公式：

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

```cpp
// ===== 一维索引：一个线程处理一个数组元素 =====
__global__ void index_demo(int* output, int n) {
    // 将 Block 坐标和线程在 Block 内的坐标合成为全局索引。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid 往往会向上取整，多出来的线程必须避免访问数组边界外。
    if (idx < n) {
        output[idx] = idx;
    }
}

int threads = 256;
// 向上取整，保证至少有 n 个线程覆盖输入数组。
int blocks = (n + threads - 1) / threads;
index_demo<<<blocks, threads>>>(d_output, n);
```

`(n + threads - 1) / threads` 是整数运算中的向上取整写法。多出来的线程必须通过 `idx < n` 进行边界保护。

### 2.4 二维和三维索引

二维矩阵中，常用一个线程负责一个 `(row, col)` 元素：

```cpp
// ===== 二维索引：一个线程处理矩阵的一个元素 =====
__global__ void matrix_set(float* matrix, int rows, int cols) {
    // x 对应列，y 对应行；这种映射适合行主序矩阵的连续列访问。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        matrix[row * cols + col] = static_cast<float>(row + col);
    }
}

dim3 threads(16, 16);
// Blocks 和 threads 都是二维布局，分别覆盖列方向和行方向。
dim3 blocks(
    (cols + threads.x - 1) / threads.x,
    (rows + threads.y - 1) / threads.y
);
matrix_set<<<blocks, threads>>>(d_matrix, rows, cols);
```

对于行主序矩阵，连续的列元素通常存放在连续地址，因此让相邻线程沿 `x/col` 方向访问，有利于 Global Memory 访存合并。

### 2.5 Block 的独立性

- 不同 Block 原则上可以独立执行。
- 一个 Block 可能先后运行在不同的 SM 上。
- 普通 Kernel 中不能直接使用 `__syncthreads()` 跨 Block 同步。
- 需要跨 Block 的全局同步时，通常拆成多个 Kernel，或使用更高级的 Cooperative Groups/Kernel Cooperative Launch。

适合 CUDA 的算法通常会尽量把协作限制在 Block 内，把 Block 之间的数据交换放到 Kernel 边界完成。

## 3. Warp 与 SIMT 执行模型

### 3.1 Warp 是硬件调度单位

GPU 会把一个 Block 中的线程划分为多个 Warp。一个 Warp 通常包含 32 个线程，但具体硬件细节应以对应架构文档为准。

```text
Block（256 threads）
├── Warp 0：thread 0  ~ 31
├── Warp 1：thread 32 ~ 63
├── ...
└── Warp 7：thread 224 ~ 255
```

线程是编程模型中的基本单位，但硬件通常以 Warp 为单位调度和发射指令。

### 3.2 SIMT 与 SIMD

CUDA 常用 **SIMT（Single Instruction, Multiple Threads）** 描述执行模型：

- 同一 Warp 的线程通常执行同一条指令。
- 每个线程有独立的寄存器和逻辑状态。
- 线程可以通过分支表达不同逻辑，但分支可能导致 Warp 分化。

SIMD 通常强调一条向量指令处理多个数据；SIMT 则让程序员以线程为单位编程，由硬件负责把线程组织为 Warp 执行。

### 3.3 Warp 分化

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx % 2 == 0) {
    output[idx] = even_path(idx);
} else {
    output[idx] = odd_path(idx);
}
```

如果同一个 Warp 内的线程同时走两个分支，硬件通常需要分别执行两个路径，再让不满足条件的线程暂时屏蔽。是否值得重写分支，应通过实际测量判断。

### 3.4 Warp 级原语

Warp 内线程可以使用 Shuffle 在寄存器之间交换数据，避免把中间值写入 Shared Memory：

```cpp
// ===== Warp 内求和：用 Shuffle 交换寄存器值 =====
__global__ void warp_sum(float* output) {
    // lane 是线程在当前 Warp 中的编号，范围通常为 0 到 31。
    int lane = threadIdx.x % 32;
    float value = static_cast<float>(lane + 1);

    for (int offset = 16; offset > 0; offset /= 2) {
        // 每轮把相邻线程的部分和相加，offset 每次减半。
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (lane == 0) {
        output[blockIdx.x] = value;
    }
}
```

`__shfl_down_sync(mask, value, delta)` 会读取同一个 Warp 中相对位置更大的线程的寄存器值。使用同步 Mask 时，应保证参与线程和 Mask 的关系正确。

### 3.5 Warp 分支和同步的关系

- Warp Shuffle 适合 Warp 内的数据交换。
- `__syncwarp()` 用于 Warp 范围的执行同步和内存顺序约束。
- `__syncthreads()` 用于同一个 Block 的线程同步。
- 跨 Block 协作通常需要 Kernel 边界、原子操作或 Cooperative Groups。

## 4. CUDA 内存层级

### 4.1 内存层级总览

```text
每个 Thread
└── Register

每个 Block
└── Shared Memory

每个 SM
└── L1 Cache / Texture Cache

整个 Device
├── L2 Cache
├── Global Memory
├── Constant Memory
└── Texture Memory
```

| 内存 | 可见范围 | 生命周期 | 特点 |
|------|----------|----------|------|
| Register | 单个 Thread | Thread 生命周期 | 延迟低，容量有限 |
| Local Memory | 单个 Thread 逻辑可见 | Thread 生命周期 | 实际位于显存，寄存器溢出时使用 |
| Shared Memory | 同一 Block | Block 生命周期 | 片上高速存储，需要同步 |
| L1 Cache | SM | 硬件管理 | 缓存部分 Global/Local 访问 |
| L2 Cache | 整个 Device | 硬件管理 | 多个 SM 共享 |
| Global Memory | 整个 Device | 应用控制 | 容量大，延迟较高 |
| Constant Memory | 整个 Device，只读 | 应用控制 | 适合少量只读常量和广播访问 |
| Texture Memory | 整个 Device，只读接口 | 应用控制 | 具有专门的缓存和寻址语义 |

### 4.2 Register

Kernel 中的普通局部变量通常优先放在 Register：

```cpp
// ===== Register：每个线程私有的局部变量 =====
__global__ void register_demo(float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f;
    float b = 2.0f;
    // a、b、c 通常优先放入 Register，线程之间互不可见。
    float c = a * b + idx;

    output[idx] = c;
}
```

Register 是每个线程私有的，线程之间不能直接读取对方的 Register。Register 使用量过高会降低一个 SM 可以同时驻留的 Warp 数量，严重时可能发生 Register Spilling。

```bash
nvcc -Xptxas=-v kernel.cu -o kernel
```

### 4.3 Global Memory 的分配与复制

```cpp
// ===== Host 与 Device Memory 的生命周期 =====
float* d_data = nullptr;
size_t bytes = n * sizeof(float);

// d_data 是 Device 指针，cudaMalloc 在显存中分配空间。
cudaMalloc(&d_data, bytes);
// H2D：把 Host 数组复制到 Device，Kernel 才能读取 d_data。
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

// 启动 Kernel 使用 d_data

// D2H：把 Kernel 产生的结果复制回 Host 数组。
cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

| 枚举 | 方向 |
|------|------|
| `cudaMemcpyHostToDevice` | Host → Device |
| `cudaMemcpyDeviceToHost` | Device → Host |
| `cudaMemcpyDeviceToDevice` | Device → Device |
| `cudaMemcpyHostToHost` | Host → Host |
| `cudaMemcpyDefault` | 由统一地址空间推断方向 |

不要把 Host 指针和 Device 指针混用。普通 `malloc` 得到的 Host 地址不能直接当作普通 Device 地址传入 Kernel。

### 4.4 Global Memory 访存合并

同一个 Warp 的线程访问 Global Memory 时，如果访问地址具有连续性，硬件可以合并内存事务。

```cpp
// 相邻线程访问相邻元素
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = input[idx];
```

```cpp
// 步长访问可能造成分散的内存事务
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = input[idx * stride];
```

常见原则：

- 让相邻线程尽量访问相邻元素。
- 二维行主序矩阵中通常让 `threadIdx.x` 映射到列。
- 避免不必要的跨大步长访问。
- 对转置、卷积和矩阵乘使用 Shared Memory 重排访问模式。

### 4.5 Shared Memory

Shared Memory 位于 SM 上，由同一 Block 的线程共享：

```cpp
// ===== Shared Memory：Block 内先加载，再重排读取 =====
__global__ void shared_demo(const float* input, float* output) {
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程把一个 Global Memory 元素搬到共享缓存。
    tile[threadIdx.x] = input[idx];

    // 确保所有线程完成写入后，才能读取其他线程写入的数据。
    __syncthreads();

    // 这里反向读取，演示 Block 内线程共享数据。
    output[idx] = tile[blockDim.x - 1 - threadIdx.x];
}
```

执行过程：

1. 每个线程把 Global Memory 的数据载入 Shared Memory。
2. 使用 `__syncthreads()` 确保整个 Block 的加载完成。
3. 线程读取 Shared Memory 中的数据并写回。

如果没有同步，某些线程可能在其他线程完成写入之前就开始读取，结果会产生未定义行为。

### 4.6 Shared Memory Bank Conflict

Shared Memory 被划分为多个 Bank。一个 Warp 的线程访问不同 Bank 时可以并行处理；如果多个线程访问同一 Bank 的不同地址，可能产生 Bank Conflict。

```text
线程访问连续地址：
thread 0 -> bank 0
thread 1 -> bank 1
thread 2 -> bank 2

线程访问冲突地址：
thread 0 -> bank 0
thread 1 -> bank 0
thread 2 -> bank 0
```

二维 Tile 转置中常见的优化方式是给第二维增加 Padding：

```cpp
__shared__ float tile[32][32 + 1];
```

Padding 改变了行跨度，可以减少某些列访问模式下的 Bank Conflict，但最终效果仍应通过 Nsight Compute 验证。

### 4.7 Constant Memory

Constant Memory 适合少量、生命周期较长且只读的数据：

```cpp
// ===== Constant Memory：多个线程读取只读系数 =====
__constant__ float coefficients[64];

__global__ void apply_coefficients(const float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 系数数组只读，适合放到 Constant Memory 中由多个线程复用。
    output[idx] = input[idx] * coefficients[idx % 64];
}
```

Host 端通过符号复制数据：

```cpp
cudaMemcpyToSymbol(coefficients, h_coefficients, sizeof(h_coefficients));
```

当同一 Warp 的线程读取同一个 Constant 地址时，硬件可以利用广播；如果线程读取大量不同地址，优势会下降。

### 4.8 Pinned Host Memory 与 Unified Memory

Pinned Host Memory 不会被操作系统换出，更适合和异步拷贝配合：

```cpp
// ===== Pinned Host Memory 配合异步拷贝 =====
float* h_data = nullptr;
cudaMallocHost(&h_data, bytes);
// Pinned Memory 不会被换出，适合 cudaMemcpyAsync 使用。
cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
cudaFreeHost(h_data);
```

Unified Memory 使用统一指针模型：

```cpp
// ===== Unified Memory：Host 和 Device 共享一个指针 =====
float* data = nullptr;
cudaMallocManaged(&data, n * sizeof(float));

for (int i = 0; i < n; ++i) {
    // CPU 可以直接写统一内存，运行时负责页面迁移。
    data[i] = static_cast<float>(i);
}

// Kernel 完成后同步，避免在 GPU 仍访问数据时释放它。
some_kernel<<<blocks, threads>>>(data, n);
cudaDeviceSynchronize();
cudaFree(data);
```

Unified Memory 简化了原型代码，但页面可能在 Host 与 Device 之间迁移。可以使用 Prefetch 提前迁移：

```cpp
cudaMemPrefetchAsync(data, bytes, device, stream);
```

## 5. 同步、异步与并发

### 5.1 Kernel 启动通常是异步的

```cpp
kernel<<<blocks, threads>>>(d_data);
std::cout << "Host continues" << std::endl;
```

Kernel 启动后，Host 线程通常不会自动等待 GPU 完成。以下操作可能建立同步边界或等待相关工作：

```cpp
cudaDeviceSynchronize();
cudaStreamSynchronize(stream);
cudaEventSynchronize(event);
cudaMemcpy(...);  // 某些同步拷贝会等待相关工作
```

为了理解程序时序，可以先显式同步；性能优化阶段再逐步引入异步和 Stream。

### 5.2 `__syncthreads()`

`__syncthreads()` 同时提供：

1. Block 内线程执行同步。
2. 一定范围内的内存可见性保证。

```cpp
__shared__ float buffer[256];

buffer[threadIdx.x] = input[idx];
__syncthreads();

float value = buffer[(threadIdx.x + 1) % blockDim.x];
```

同一个 Block 中参与执行的线程必须能够到达同一个同步点，不要把它放在只有部分线程会进入的分支中：

```cpp
// 错误示例：部分线程到达同步点，可能导致 Kernel 无法结束
if (threadIdx.x < 32) {
    __syncthreads();
}
```

### 5.3 原子操作

当多个线程需要更新同一个地址时，可以使用原子操作：

```cpp
__global__ void histogram(const int* input, int* bins, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = input[idx];
        atomicAdd(&bins[bin], 1);
    }
}
```

原子操作能保护更新的正确性，但高竞争地址会串行化。常见优化路径是每个 Block 先构建局部 Histogram，再把局部结果合并到全局结果。

### 5.4 Stream

Stream 是按顺序执行的一组 GPU 操作。不同 Stream 中没有依赖的操作可能并发：

```cpp
// ===== Stream 中的操作按提交顺序执行 =====
cudaStream_t stream_a;
cudaStream_t stream_b;
cudaStreamCreate(&stream_a);
cudaStreamCreate(&stream_b);

cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream_a);
cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream_b);

kernel_a<<<blocks, threads, 0, stream_a>>>(d_a);
kernel_b<<<blocks, threads, 0, stream_b>>>(d_b);

// 等待各自 Stream 中的拷贝和 Kernel 完成，再销毁 Stream。
cudaStreamSynchronize(stream_a);
cudaStreamSynchronize(stream_b);

cudaStreamDestroy(stream_a);
cudaStreamDestroy(stream_b);
```

并发是否发生取决于 GPU 是否支持相应的 Copy Engine 和 Kernel 并发、Host Memory 是否适合异步传输、操作之间是否存在依赖，以及资源使用量是否允许多个 Kernel 同时驻留。

### 5.5 Event 与 GPU 计时

GPU Event 用于记录 Stream 中的时间点，适合测量 GPU 工作时长：

```cpp
// ===== 用 CUDA Event 测量 GPU 时间 =====
cudaEvent_t start;
cudaEvent_t stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// start/stop 记录的是同一 Stream 中 GPU 操作的时间点。
kernel<<<blocks, threads>>>(d_data);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("kernel: %.3f ms\\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

不要只用 CPU 的 `std::chrono` 包围异步 Kernel 启动：

```cpp
// 这种写法只测到了提交开销，不能代表 Kernel 执行时间
auto begin = std::chrono::high_resolution_clock::now();
kernel<<<blocks, threads>>>(d_data);
auto end = std::chrono::high_resolution_clock::now();
```

如果必须使用 CPU 计时，至少要在结束前同步：

```cpp
auto begin = std::chrono::high_resolution_clock::now();
kernel<<<blocks, threads>>>(d_data);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
```

### 5.6 重叠数据传输和计算

将数据拆成多个 Chunk，可以尝试让传输和计算重叠：

```text
时间轴 ───────────────────────────────────►
Stream 0: H2D(chunk 0) | Kernel(chunk 0) | D2H(chunk 0)
Stream 1:              H2D(chunk 1) | Kernel(chunk 1) | D2H(chunk 1)
```

实际实现通常需要 Pinned Host Memory、多个非默认 Stream、明确的 Event 依赖和合适的 Chunk 大小。

## 6. 常见并行模式

### 6.1 向量加法

#### 计算逻辑

$$
C_i = A_i + B_i
$$

每个线程负责一个索引：

```cpp
// ===== 向量加法：线程索引直接映射到数据索引 =====
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    // 一个线程只负责计算一个 c[idx]，因此线程之间没有写冲突。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

#### 启动方式

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

向量加法通常是 Memory-Bound Kernel，性能主要受访存带宽和数据传输影响，单纯增加算术操作并不会自动提高吞吐。

### 6.2 并行归约

归约将多个元素聚合为一个结果，例如求和：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
          │ 两两相加
          ▼
[3, 7, 11, 15]
          │
          ▼
[10, 26]
          │
          ▼
[36]
```

#### 基础 Shared Memory 版本

```cpp
// ===== 归约：每个 Block 把局部和写入一个输出元素 =====
__global__ void reduce_sum_basic(const float* input, float* output, int n) {
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 不足一个完整 Block 的尾部元素用 0 补齐。
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            // 当前线程把另一半区间的部分和累加到自己所在位置。
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}
```

动态 Shared Memory 的大小通过第三个启动参数传入：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
reduce_sum_basic<<<blocks, threads, threads * sizeof(float)>>>(
    d_input,
    d_partial,
    n
);
```

这只完成了每个 Block 的局部归约。多个 Block 的结果还需要再次归约，可以重复启动 Kernel，或使用 CUB 等库提供的实现。

#### 归约中的优化方向

- 使用连续访问，减少 Global Memory 事务。
- 使用交错寻址减少早期分支分化。
- 在最后一个 Warp 中使用 Shuffle，减少同步和 Shared Memory 访问。
- 让每个线程处理多个元素，降低启动和索引开销。
- 使用 CUB 的 `BlockReduce` 或 `DeviceReduce` 进行成熟实现。

### 6.3 分块矩阵乘法

矩阵乘法：

$$
C_{row,col} = \sum_{k=0}^{K-1} A_{row,k}B_{k,col}
$$

朴素版本中，不同线程会重复从 Global Memory 读取相同的 A/B 元素。Tile 版本让一个 Block 协同加载一小块数据到 Shared Memory：

```text
Global Memory
     │ 协同加载
     ▼
Shared A Tile + Shared B Tile
     │ Block 内重复使用
     ▼
寄存器累加 C 的局部结果
```

```cpp
// ===== Tile 矩阵乘：Block 协作加载并重复使用数据 =====
template <int TILE>
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    // 每个线程最终负责 C 中一个 row/col 位置的累加。
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float value = 0.0f;

    for (int tile_index = 0; tile_index < (K + TILE - 1) / TILE; ++tile_index) {
        int a_col = tile_index * TILE + threadIdx.x;
        int b_row = tile_index * TILE + threadIdx.y;

        // 越过矩阵边界的位置补 0，保证任意尺寸都能运行。
        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // 所有线程共享当前 Tile，并在寄存器 value 中累加内积。
        for (int k = 0; k < TILE; ++k) {
            value += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}
```

```cpp
constexpr int TILE = 16;
dim3 threads(TILE, TILE);
dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
matmul_tiled<TILE><<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);
```

分块矩阵乘需要同时考虑 Global Memory 加载是否合并、Tile 大小是否造成过高 Shared Memory 使用量、累加值是否造成寄存器压力，以及与 cuBLAS 的实现相比手写 Kernel 是否真的有必要。

### 6.4 矩阵转置

直接转置可能造成写入不连续：

```cpp
output[col * rows + row] = input[row * cols + col];
```

一种常见思路是使用 Shared Memory Tile：

```cpp
// ===== Tile 转置：把不连续访问转换为共享内存访问 =====
__global__ void transpose_tiled(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float tile[32][32 + 1];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 第一阶段按行连续读取 Global Memory。
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    int transposed_x = blockIdx.y * 32 + threadIdx.x;
    int transposed_y = blockIdx.x * 32 + threadIdx.y;

    // 第二阶段交换行列索引，按输出布局写回。
    if (transposed_x < rows && transposed_y < cols) {
        output[transposed_y * rows + transposed_x] =
            tile[threadIdx.x][threadIdx.y];
    }
}
```

`32 + 1` 的 Padding 用于缓解转置读取时的 Shared Memory Bank Conflict。

### 6.5 Histogram 与原子操作

Histogram 的并行难点不是读取，而是多个线程可能同时更新相同的 Bin：

```cpp
// ===== 用原子操作统计 Histogram =====
__global__ void histogram_global(
    const unsigned char* input,
    unsigned int* bins,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 多个线程可能命中同一个 bin，因此更新必须是原子的。
        atomicAdd(&bins[input[idx]], 1);
    }
}
```

当数据分布集中、竞争严重时，可以改成：

1. 每个 Block 在 Shared Memory 中维护局部 Bin。
2. Block 内线程完成局部统计。
3. 使用原子操作把局部 Bin 合并到 Global Memory。

### 6.6 Prefix Sum、Scan 与 Stream Compaction

```text
输入：  [3, 1, 7, 0]
输出：  [3, 4, 11, 11]
```

Scan 是很多并行算法的基础，可用于根据标记压缩有效元素、构造输出位置、基数排序和分桶。完整 Scan 需要处理上扫、下扫、Block 间部分和等问题，实际开发通常优先使用 CUB 的 `DeviceScan`。

## 7. 性能分析与优化

### 7.1 先确认正确性，再测量性能

```text
建立正确的 CPU Reference
          │
          ▼
验证 GPU 输出和 Reference 一致
          │
          ▼
构造具有代表性的输入规模
          │
          ▼
使用 GPU Event 或 Profiler 测量
          │
          ▼
定位瓶颈：传输、计算、访存、同步或资源限制
          │
          ▼
一次只修改一个主要因素
          │
          ▼
重新验证正确性和性能
```

GPU 程序的总耗时至少包括：

```text
总时间 = Host 准备 + H2D 传输 + Kernel 执行 + D2H 传输 + 同步等待
```

### 7.2 Memory-Bound 与 Compute-Bound

Memory-Bound Kernel 的主要瓶颈是数据搬运，常见于向量加法和简单逐元素变换；Compute-Bound Kernel 的主要瓶颈是算术或特殊函数吞吐，常见于复杂矩阵计算。

算术强度：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes moved}}
$$

### 7.3 访存优化

1. 减少 Host 与 Device 之间的数据传输。
2. 让 Warp 的 Global Memory 访问尽可能合并。
3. 消除重复的 Global Memory 读取。
4. 使用 Shared Memory 或 Register 重用数据。
5. 检查 Shared Memory Bank Conflict。
6. 确认缓存和数据布局是否适合访问模式。

### 7.4 Occupancy

Occupancy 通常表示一个 SM 上活跃 Warp 数与硬件最大可支持 Warp 数的比例：

$$
\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}
$$

影响 Occupancy 的资源包括每个 Block 的线程数量、每个线程的 Register 使用量、每个 Block 的 Shared Memory 使用量，以及 SM 支持的最大线程数和最大 Block 数。

Occupancy 高通常有利于隐藏内存延迟，但不是越高越好。过度追求 Occupancy 可能增加寄存器溢出、降低单线程指令级并行度，最终应结合实际 Kernel 时间和 Profiler 指标判断。

### 7.5 Block 大小

常见起点是使用 128 或 256 个线程，并保持线程数为 Warp 大小的整数倍：

```cpp
int candidates[] = {128, 256, 512};
```

选择时需要考虑算法是否频繁同步、Shared Memory 和 Register 使用量、边界线程占比、Kernel 的访存和计算特征，以及 GPU 架构限制。

### 7.6 Register Spilling

当 Kernel 需要的 Register 超过可分配范围时，部分局部变量可能被放入 Local Memory。Local Memory 逻辑上属于线程私有数据，但实际位于 Device Memory，访问代价会明显增加。

```bash
nvcc -Xptxas=-v kernel.cu -o kernel
```

如果寄存器使用量过高，可以减少过大的局部数组、降低同时存活的临时变量、调整循环展开程度，或让数据分批处理。

### 7.7 Kernel Launch 配置实验

```cpp
for (int threads : {64, 128, 256, 512}) {
    int blocks = (n + threads - 1) / threads;

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("threads=%d, time=%.3f ms\\n", threads, ms);
}
```

实验应记录 GPU 型号、计算能力、CUDA Toolkit、编译参数、输入尺寸、Warmup 次数、重复次数，以及是否包含 H2D/D2H 传输。

### 7.8 Nsight Systems

Nsight Systems 适合查看完整程序时间线：

```bash
nsys profile -o cuda_timeline ./vector_add
nsys stats cuda_timeline.nsys-rep
```

重点观察 CPU 是否等待 GPU、H2D/Kernel/D2H 是否串行、多 Stream 是否产生并发、哪些 Kernel 占据主要时间，以及是否存在过多短 Kernel 启动。

### 7.9 Nsight Compute

Nsight Compute 用于分析单个 Kernel：

```bash
ncu --set full -o kernel_report ./vector_add
```

常见关注指标：Achieved Occupancy、Global Memory Load/Store Efficiency、DRAM Throughput、L1/L2 Cache 命中情况、Shared Memory Bank Conflicts、Warp Stall Reasons，以及指令吞吐和算术管线利用率。

## 8. 错误检查与调试

### 8.1 CUDA API 错误检查宏

CUDA Runtime API 函数通常返回 `cudaError_t`。建议检查每一次关键调用：

```cpp
// ===== 统一检查 CUDA Runtime API 返回值 =====
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        /* 先执行调用，再把错误码转换成可读信息。 */                         \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(                                                   \
                stderr,                                                      \
                "CUDA error at %s:%d: %s\\n",                              \
                __FILE__,                                                    \
                __LINE__,                                                    \
                cudaGetErrorString(error)                                   \
            );                                                               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
```

使用：

```cpp
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
```

### 8.2 Kernel 启动错误与执行错误

```cpp
// ===== 分别检查 Kernel 启动错误和异步执行错误 =====
kernel<<<blocks, threads>>>(d_input, d_output, n);

// 检查启动配置、参数和资源限制等立即可见的错误。
CUDA_CHECK(cudaGetLastError());

// 等待 Kernel 完成，并检查异步执行期间产生的错误。
CUDA_CHECK(cudaDeviceSynchronize());
```

`cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的作用不同。只调用前者，可能漏掉异步执行期间出现的错误；只调用后者，则不容易区分是启动错误还是执行错误。

### 8.3 常见错误类型

| 错误 | 可能原因 |
|------|----------|
| `invalid configuration argument` | Block 维度、Grid 维度或动态 Shared Memory 参数不合法 |
| `too many resources requested for launch` | Register、Shared Memory 或线程资源超出限制 |
| `illegal memory access` | 数组越界、错误指针或错误地址空间 |
| `misaligned address` | 指针对齐要求不满足 |
| `an illegal instruction was encountered` | 二进制架构或代码执行状态不匹配 |
| Kernel 无法结束 | 同步点使用不一致、死循环或等待条件错误 |

### 8.4 Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./vector_add
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool synccheck ./program
compute-sanitizer --tool initcheck ./program
```

分别用于检查非法内存访问、竞争条件、同步问题和未初始化数据。适合先使用较小输入运行 Sanitizer，确认正确性后再恢复完整规模。

### 8.5 `printf` 调试

```cpp
// ===== 只限制少量线程进行 Device printf 调试 =====
__global__ void debug_kernel(const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x == 0 && threadIdx.x < 4) {
        printf("idx=%d value=%f\\n", idx, input[idx]);
    }
}
```

调试后应删除或限制输出，因为大量 Device `printf` 会显著影响性能，甚至占满输出缓冲区。

### 8.6 浮点数与归约结果

浮点加法不满足严格结合律：

```text
(a + b) + c  可能不等于  a + (b + c)
```

并行归约改变了加法顺序，因此 GPU 结果和 CPU 顺序累加结果可能存在微小误差。验证时应使用相对误差或绝对误差：

```cpp
bool close_enough(float actual, float expected) {
    float absolute_error = std::abs(actual - expected);
    float relative_error = absolute_error / std::max(std::abs(expected), 1.0f);
    return absolute_error < 1e-5f || relative_error < 1e-5f;
}
```

## 9. CUDA 库与并行抽象

### 9.1 为什么优先考虑已有库

高质量 CUDA 库通常已经处理了多种 GPU 架构、边界条件、数据类型、复杂的 Tile/ Warp / 异步拷贝策略、算法选择和性能调优。因此，手写 Kernel 前应先确认是否有适合的库实现。

### 9.2 cuBLAS

矩阵乘法可以使用 cuBLAS：

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

const float alpha = 1.0f;
const float beta = 0.0f;

cublasSgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    n,
    m,
    k,
    &alpha,
    d_b,
    n,
    d_a,
    k,
    &beta,
    d_c,
    n
);

cublasDestroy(handle);
```

cuBLAS 默认使用列主序语义。和 C/C++ 中常见的行主序数组交互时，需要通过转置参数、数据布局或矩阵维度映射避免结果错误。

### 9.3 Thrust

Thrust 提供类似 STL 的并行容器和算法：

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

thrust::device_vector<float> values(n, 1.0f);
float result = thrust::reduce(values.begin(), values.end(), 0.0f);
```

Thrust 适合快速表达常见算法；当需要精细控制内存、Stream 或 Kernel 配置时，可以进一步使用 CUB 或自定义 Kernel。

### 9.4 CUB

CUB 提供 Block、Warp 和 Device 级别的并行原语，例如 `BlockReduce`、`BlockScan`、`BlockLoad`、`BlockStore`、`DeviceReduce` 和 `DeviceScan`。

```cpp
using BlockReduce = cub::BlockReduce<float, 256>;
__shared__ typename BlockReduce::TempStorage temp_storage;

float block_sum = BlockReduce(temp_storage).Sum(thread_value);
```

不同 CUDA Toolkit 版本中 CUB 的头文件组织和集成方式可能变化，实际编译时以当前 Toolkit 文档为准。

## 10. PyTorch 与 CUDA

### 10.1 Tensor 设备管理

```python
# ===== 在 Python 端选择 CPU 或 CUDA 设备 =====
import torch

# device 是后续 Tensor 创建和计算所使用的目标设备。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1024, device=device)
y = torch.randn(1024, device=device)
z = x + y

# z 会继承 x 和 y 的 CUDA 设备位置。
print(z.device)
```

常见方式：

```python
x = x.cuda()
x = x.to("cuda")
x = x.to(device)
```

不要在 CPU Tensor 和 CUDA Tensor 之间直接进行运算：

```python
cpu_tensor = torch.randn(4)
gpu_tensor = torch.randn(4, device="cuda")

# 会产生设备不一致错误
# result = cpu_tensor + gpu_tensor
```

### 10.2 PyTorch 中的异步执行与计时

PyTorch CUDA 操作通常也具有异步性质，测量 Kernel 时间时需要同步：

```python
# ===== 用同步边界测量 CUDA 矩阵乘 =====
import time
import torch

x = torch.randn(4096, 4096, device="cuda")
y = torch.randn(4096, 4096, device="cuda")

for _ in range(5):
    # Warmup 让首次初始化和内核选择成本不计入正式测量。
    _ = x @ y

torch.cuda.synchronize()
begin = time.perf_counter()

for _ in range(20):
    _ = x @ y

torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - begin) * 1000 / 20
print(f"average: {elapsed_ms:.3f} ms")
```

更适合 GPU 计时的方式是 `torch.cuda.Event`：

```python
# ===== 用 CUDA Event 记录 GPU 时间 =====
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = x @ y
end.record()

end.synchronize()
print(f"GPU time: {start.elapsed_time(end):.3f} ms")
```

### 10.3 PyTorch 显存管理

```python
# ===== 查看 PyTorch CUDA 缓存分配器的状态 =====
if torch.cuda.is_available():
    print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
```

两个指标的区别：

- `memory_allocated()`：当前 Tensor 实际占用的显存。
- `memory_reserved()`：PyTorch 缓存分配器向 CUDA 保留的显存。

`torch.cuda.empty_cache()` 只能释放缓存分配器中暂时没有使用的块，不能释放仍被 Tensor 引用的显存：

```python
# ===== 释放 Python 引用后清理暂未使用的缓存块 =====
del x, y, z
torch.cuda.empty_cache()
```

### 10.4 自定义 CUDA Extension 的基本结构

```text
Python API
    │
    ▼
PyBind11 / torch.library 绑定
    │
    ▼
C++ Host 包装函数
    │
    ▼
CUDA Kernel
```

```cpp
// my_op.cpp
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "My CUDA forward");
}
```

```cpp
// ===== CUDA Extension 的 Kernel 入口和边界检查 =====
// my_op.cu
#include <torch/extension.h>

__global__ void identity_kernel(
    const float* input,
    float* output,
    int n
) {
    // data_ptr<float>() 把 Tensor 的底层存储交给 CUDA Kernel 使用。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int n = static_cast<int>(input.numel());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    identity_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}
```

实际扩展还需要处理当前 CUDA Stream、非连续 Tensor、dtype/维度/设备检查、大于 `int` 范围的 Tensor 元素数量、Kernel Launch 错误，以及 C++/CUDA/PyTorch ABI 兼容性。

### 10.5 Numba CUDA

Numba 可以用 Python 装饰器编写简单 CUDA Kernel：

```python
import math

import numpy as np

try:
    from numba import cuda
except ImportError:
    cuda = None

if cuda is not None and cuda.is_available():
    @cuda.jit
    def add_kernel(a, b, c):
        idx = cuda.grid(1)
        if idx < c.size:
            c[idx] = a[idx] + b[idx]

    n = 1 << 20
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)

    threads = 256
    blocks = math.ceil(n / threads)
    add_kernel[blocks, threads](a, b, c)
    print(np.max(np.abs(c - 2.0)))
else:
    print("Numba CUDA 不可用，跳过示例")
```

Numba 适合验证索引映射和简单 Kernel；复杂算子、精细性能控制和框架集成通常需要 CUDA C++、CUDA 库或其他 Kernel DSL。

### 10.6 CUDA、Triton 与高层库的关系

```text
cuBLAS / cuDNN / CUB
        │ 直接复用成熟实现
        ▼
PyTorch / JAX 等框架
        │ 调用 CUDA 库或自定义后端
        ▼
Triton / CUDA C++
        │ 编写自定义 GPU Kernel
        ▼
CUDA Runtime / Driver
        │ 设备管理、内存、执行和同步
        ▼
GPU 硬件
```

| 需求 | 优先考虑 |
|------|----------|
| 标准矩阵乘、卷积、FFT | cuBLAS、cuDNN、cuFFT |
| 标准归约、Scan、排序 | CUB、Thrust |
| 快速验证简单 Kernel | Numba CUDA、Triton |
| 需要完全控制底层执行 | CUDA C++ |
| 常见 Tensor 运算 | PyTorch 内置算子 |

## 11. 一个完整的 CUDA 学习实验

### 11.1 实验目标

实现并验证向量加法，分别观察 Host 数据准备、Device Memory 分配、H2D 和 D2H 复制、Kernel 启动、CUDA Event 计时、错误检查和 CPU Reference 对比。

### 11.2 实验代码骨架

```cpp
// ===== 完整实验：Host 准备、拷贝、Kernel、校验和释放 =====
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s\\n", cudaGetErrorString(error));     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    // 每个线程计算一个输出元素，n 负责防止尾部线程越界。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const size_t bytes = n * sizeof(float);

    // h_* 位于 Host，d_* 位于 Device；两类指针不能混用。
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 向上取整的 blocks 保证所有 n 个元素都有线程覆盖。
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // CPU Reference 预期每个元素为 1 + 2 = 3。
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::abs(h_c[i] - 3.0f));
    }
    std::printf("max error: %.6f\\n", max_error);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return max_error < 1e-5f ? 0 : 1;
}
```

编译运行：

```bash
nvcc -O2 -lineinfo vector_add.cu -o vector_add
./vector_add
```

Windows 下：

```powershell
nvcc -O2 -lineinfo vector_add.cu -o vector_add.exe
.\\vector_add.exe
```

### 11.3 进一步实验

在保持结果一致的前提下，依次修改：

1. `threads` 为 64、128、256、512。
2. 输入规模为 `1 << 10`、`1 << 20`、`1 << 26`。
3. 使用 Pageable 和 Pinned Host Memory。
4. 用 CPU 计时与 CUDA Event 计时分别测量。
5. 使用 `compute-sanitizer` 运行。
6. 使用 Nsight Systems 查看 H2D、Kernel、D2H 的时间线。

## 12. 常用命令速查

### 12.1 环境和编译

```bash
nvidia-smi
nvcc --version
nvcc -std=c++17 -O3 kernel.cu -o kernel
nvcc -Xptxas=-v kernel.cu -o kernel
```

### 12.2 运行和错误检查

```bash
./kernel
compute-sanitizer --tool memcheck ./kernel
compute-sanitizer --tool racecheck ./kernel
compute-sanitizer --tool synccheck ./kernel
```

### 12.3 性能分析

```bash
nsys profile -o report ./kernel
nsys stats report.nsys-rep
ncu --set full -o kernel_report ./kernel
```

### 12.4 Python 端检查

```python
import torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
```

## 13. 学习顺序

### 13.1 第一阶段：能读懂并运行 Kernel

1. Host/Device 和 CUDA Runtime。
2. `__global__`、Kernel 启动和错误检查。
3. `threadIdx`、`blockIdx`、`blockDim`、`gridDim`。
4. 一维向量加法和二维矩阵索引。
5. `cudaMalloc`、`cudaMemcpy`、`cudaFree`。

### 13.2 第二阶段：理解线程协作

1. Warp、SIMT 和分支分化。
2. Shared Memory 和 `__syncthreads()`。
3. 原子操作和竞争条件。
4. Warp Shuffle。
5. 归约、Histogram 和转置。

### 13.3 第三阶段：理解性能

1. CUDA Event 和稳定的 Benchmark。
2. Global Memory 访存合并。
3. Shared Memory Bank Conflict。
4. Register、Local Memory 和 Spilling。
5. Occupancy、Stream 和数据传输重叠。
6. Nsight Systems、Nsight Compute 和 Compute Sanitizer。

### 13.4 第四阶段：连接框架和库

1. cuBLAS、cuDNN、Thrust、CUB。
2. PyTorch CUDA Tensor 和 Stream。
3. PyTorch C++/CUDA Extension。
4. Numba CUDA 或 Triton Kernel。
5. 对比手写 Kernel、库实现和框架算子。

## 14. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Runtime API Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)
- [CUDA Runtime API Events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUB Documentation](https://nvidia.github.io/cccl/cub/)
