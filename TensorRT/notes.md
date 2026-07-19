# TensorRT 推理部署

TensorRT 是 NVIDIA 面向深度学习推理的优化 SDK。它接收已经训练好的模型或 ONNX 图，在构建阶段为目标 GPU 选择合适的算子实现、数据格式和执行计划，生成可序列化的 TensorRT engine，随后由 Runtime 加载并执行推理。

TensorRT 的核心对象不是训练过程，而是：

```text
训练框架模型
  -> ONNX 或 TensorRT Network Definition
  -> Builder + BuilderConfig
  -> Serialized Engine / Plan
  -> Runtime + ExecutionContext
  -> GPU 推理
```

## 内容概览

- TensorRT 的工作模型和核心对象
- 安装、环境检查和版本关系
- PyTorch 导出 ONNX、检查 ONNX、构建 Engine
- `trtexec` 的构建、推理和性能测量
- TensorRT Python Runtime API
- 静态形状、动态形状和 Optimization Profile
- FP32、TF32、FP16、BF16、INT8 与精度验证
- Layer Fusion、Tactic、Workspace、Timing Cache 和 CUDA Graph
- Plugin、Engine Inspector、错误检查和性能分析
- TensorRT、Torch-TensorRT、TensorRT-LLM 和 Triton 的边界

> 说明：TensorRT 的 API、支持矩阵和命令行参数会随版本变化。本文解释稳定的概念，并以当前官方 Python API 中的 Tensor 地址绑定和 `execute_async_v3` 方式作为示例；运行前应使用本机版本的 `trtexec --help` 和官方文档核对细节。

## 1. TensorRT 的基本工作模型

### 1.1 推理与训练

**训练（training）**通过数据和损失函数更新模型参数；**推理（inference）**使用已经固定的参数计算新输入的输出。TensorRT 主要优化推理阶段，因此它通常不会替代 PyTorch、TensorFlow 等训练框架。

一个训练框架中的模型包含 Python 控制逻辑、参数和算子；TensorRT Engine 则是面向特定运行环境编译后的执行产物。Engine 通常包含权重、网络结构信息、选定的 Tactic 和内存规划。

### 1.2 Builder、Engine 和 Runtime

| 对象 | 作用 | 所处阶段 |
|---|---|---|
| `ILogger` / `Logger` | 接收构建和运行时日志 | 构建、加载、执行 |
| `IBuilder` / `Builder` | 编译网络并生成 Engine | 离线构建 |
| `INetworkDefinition` | 保存 TensorRT 网络图 | 构建 |
| `OnnxParser` | 将 ONNX 转换为 TensorRT 网络图 | 构建 |
| `IBuilderConfig` / `BuilderConfig` | 设置精度、内存池、Profile 等构建选项 | 构建 |
| Serialized Engine / Plan | 序列化后的硬件相关二进制 | 构建产物 |
| `IRuntime` / `Runtime` | 反序列化 Engine | 运行时 |
| `ICudaEngine` / `Engine` | 保存编译后的网络和 I/O 元数据 | 运行时 |
| `IExecutionContext` / `ExecutionContext` | 保存一次推理执行所需的状态 | 运行时 |

**Builder** 通常在部署前离线运行，因为它需要搜索和比较多种实现。**Runtime** 的职责是加载已经构建好的 Engine；**ExecutionContext** 才是实际提交一次推理的对象。

### 1.3 Engine、Plan、ONNX 的关系

- **ONNX**：跨框架的模型交换格式，描述图结构、权重和张量信息。
- **Network Definition**：TensorRT 内部的网络图表示，可以由 ONNX Parser 导入，也可以使用 TensorRT API 手动构建。
- **Engine / Plan**：Builder 根据网络、精度、Profile、目标 GPU 和构建配置生成的序列化执行计划。

ONNX 更适合作为交换和检查格式；Engine 更适合作为部署运行格式。Engine 不是通用模型文件，不能默认在任意 GPU、驱动、TensorRT 版本或操作系统上直接复用。

### 1.4 Engine 构建和推理的最小流程

```text
构建阶段：
Logger -> Builder -> Network -> Parser -> BuilderConfig -> Serialized Engine

推理阶段：
Logger -> Runtime -> Engine -> ExecutionContext
       -> 设置输入形状和设备地址 -> enqueue -> 同步并读取输出
```

理解这两条链路很重要：构建阶段决定“怎样执行”，运行阶段负责“按这个计划执行”。

## 2. 安装与环境检查

### 2.1 需要哪些组件

一个完整的 TensorRT 开发环境通常包括：

- NVIDIA Driver：让操作系统访问 GPU。
- CUDA Runtime / Toolkit：提供 GPU 运行时、编译器和 CUDA 库。
- TensorRT Runtime：加载和执行 Engine。
- TensorRT Python 或 C++ API：构建和集成应用。
- ONNX Parser：把 ONNX 图导入 TensorRT。
- `trtexec`：构建 Engine、运行推理和测量性能的命令行工具。

如果只需要运行已构建的 Engine，可以使用更精简的 Runtime；如果需要构建 Engine，则需要 Builder 相关库和工具。

### 2.2 查询驱动、CUDA 和 TensorRT

```bash
# 查看 GPU、驱动和驱动声明支持的 CUDA 上限
nvidia-smi

# 查看本机 Toolkit 中的 nvcc 版本
nvcc --version

# 查看 TensorRT 命令行工具是否可用
trtexec --help
```

`nvidia-smi` 显示的是驱动层信息，`nvcc --version` 显示的是 Toolkit 编译器信息，二者不一定相同。`trtexec --help` 是确认 TensorRT 命令行安装和当前参数集合的直接方式。

### 2.3 使用 Python 检查绑定

```python
# ===== 检查 TensorRT Python 绑定和 CUDA 设备 =====
import tensorrt as trt

# 版本号用于记录 API 和 Engine 的构建环境。
print("TensorRT:", trt.__version__)

# Logger 控制 TensorRT 输出到终端的日志级别。
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

print("platform has fast FP16:", builder.platform_has_fast_fp16)
print("platform has fast INT8:", builder.platform_has_fast_int8)
```

不同 TensorRT 版本和平台的属性支持可能不同。遇到属性不存在时，应先查对应版本的 Python API，而不是把不同版本的代码混用。

### 2.4 版本关系

构建和运行时至少需要关注：

| 维度 | 影响 |
|---|---|
| GPU 架构 / Compute Capability | 决定可用的数据格式和 Kernel |
| NVIDIA Driver | 决定 CUDA 驱动接口和设备访问能力 |
| CUDA Runtime / Toolkit | 影响运行、编译和扩展链接 |
| TensorRT 版本 | 影响 API、算子支持和 Engine 兼容性 |
| 操作系统与平台 | 影响库文件、Python 绑定和支持范围 |

建议在构建 Engine 时保存环境信息、输入形状、精度设置和构建命令。部署环境发生变化时，应重新确认兼容性并重新测量。

## 3. ONNX 模型准备

### 3.1 为什么经常从 ONNX 开始

ONNX 能把训练框架中的模型图导出为相对独立的交换格式。TensorRT 可以通过 ONNX Parser 将图导入网络定义，再继续执行构建流程。

从 ONNX 开始并不意味着所有模型都能直接转换。模型中的每个算子、数据类型、形状关系和控制流都必须能被 TensorRT 表达；不支持的部分需要改写、拆分或使用 Plugin。

### 3.2 导出一个静态形状 ONNX

```python
# ===== 将 PyTorch 模型导出为静态形状 ONNX =====
import torch


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        # 输入形状为 [batch, 3, 32, 32]，输出形状为 [batch, 10]。
        return self.fc(self.flatten(x))


model = TinyClassifier().eval()
dummy_input = torch.randn(1, 3, 32, 32)

# input_names/output_names 让后续 trtexec 和 Runtime 代码可以稳定引用张量名。
torch.onnx.export(
    model,
    (dummy_input,),
    "tiny_classifier.onnx",
    input_names=["images"],
    output_names=["logits"],
    opset_version=17,
)
```

`dummy_input` 不负责训练，它只用于让导出器执行一次前向传播并记录图结构。导出后应确认输入输出名称、数据类型和维度与预期一致。

### 3.3 导出动态 batch 的 ONNX

```python
# ===== 只让 batch 维度动态，保持图像空间尺寸固定 =====
torch.onnx.export(
    model,
    (dummy_input,),
    "tiny_classifier_dynamic.onnx",
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={
        "images": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=17,
)
```

**静态维度**在构建时已经确定；**动态维度**在 ONNX 中用符号或 `-1` 表示，运行时再提供实际数值。动态形状增加了灵活性，也要求 TensorRT 在构建时得到每个动态输入的范围。

### 3.4 检查 ONNX 图

```bash
# 使用 ONNX checker 检查模型结构和基本合法性
python -c "import onnx; onnx.checker.check_model(onnx.load('tiny_classifier.onnx'))"
```

也可以使用 Netron、ONNX Runtime 或 `onnx.shape_inference` 观察节点、输入输出和中间形状：

```python
# ===== 用 ONNX Python API 查看输入输出 =====
import onnx

model = onnx.load("tiny_classifier.onnx")
onnx.checker.check_model(model)

for value in model.graph.input:
    print("input:", value.name, value.type.tensor_type.shape)

for value in model.graph.output:
    print("output:", value.name, value.type.tensor_type.shape)
```

### 3.5 ONNX 导出检查清单

- 输入输出名称是否稳定。
- 输入输出数据类型是否符合 Runtime 代码。
- 静态和动态维度是否符合实际调用范围。
- 是否存在未初始化参数或训练态算子。
- ONNX Runtime 与原始框架输出是否接近。
- TensorRT Parser 是否能够识别所有节点。

## 4. 使用 `trtexec` 构建和运行

### 4.1 `trtexec` 的定位

`trtexec` 是 TensorRT 自带的命令行工具，可以：

1. 从 ONNX 构建 Engine。
2. 加载已经序列化的 Engine。
3. 执行推理并统计延迟、吞吐和显存信息。
4. 输出层级信息并配合 Nsight 工具分析。

它适合先验证模型和环境，再编写自己的 Python/C++ 集成代码。

### 4.2 从静态 ONNX 构建 Engine

```bash
# 从 ONNX 导入网络并保存 TensorRT plan
trtexec \
  --onnx=tiny_classifier.onnx \
  --saveEngine=tiny_classifier.plan

# 允许 Builder 选择 FP16 实现；不代表每一层都一定使用 FP16
trtexec \
  --onnx=tiny_classifier.onnx \
  --saveEngine=tiny_classifier_fp16.plan \
  --fp16
```

**精度标志是允许（permission），不是绝对命令。** Builder 会在满足约束的前提下选择更快的实现；某些层可能仍使用更高精度。

### 4.3 从动态 ONNX 构建 Engine

动态 batch 的 ONNX 需要给出输入的最小、最佳和最大形状：

```bash
trtexec \
  --onnx=tiny_classifier_dynamic.onnx \
  --minShapes=images:1x3x32x32 \
  --optShapes=images:4x3x32x32 \
  --maxShapes=images:8x3x32x32 \
  --saveEngine=tiny_classifier_dynamic.plan
```

**Optimization Profile** 是一组动态输入范围：

| 形状 | 含义 |
|---|---|
| `min` | 允许构建和执行的最小形状 |
| `opt` | Builder 重点优化的代表形状 |
| `max` | 允许构建和执行的最大形状 |

运行时给定的输入形状必须落在某个 Profile 覆盖范围内。`opt` 不是唯一可运行的形状，但通常是 Builder 重点调优的位置。

### 4.4 加载 Engine 测量推理

```bash
# 只加载 Engine，不在本次命令中重新构建
trtexec \
  --loadEngine=tiny_classifier_fp16.plan \
  --shapes=images:1x3x32x32 \
  --warmUp=500 \
  --duration=10
```

常见参数：

| 参数 | 用途 |
|---|---|
| `--onnx` | 指定 ONNX 输入 |
| `--saveEngine` | 保存构建后的 Engine |
| `--loadEngine` | 加载已有 Engine |
| `--fp16` | 允许 FP16 实现 |
| `--int8` | 允许 INT8 实现，具体量化信息仍需满足模型要求 |
| `--shapes` | 为动态输入指定本次运行形状 |
| `--minShapes/--optShapes/--maxShapes` | 创建动态形状 Profile |
| `--warmUp` | 设置最短预热时间 |
| `--duration` | 设置最短测量时间 |
| `--iterations` | 设置最小测量次数 |
| `--noDataTransfers` | 排除 Host 与 Device 数据传输，观察 GPU 推理部分 |
| `--dumpLayerInfo` | 输出层级信息 |
| `--profilingVerbosity=detailed` | 增加可用于分析的层级信息 |

完整参数以当前版本的 `trtexec --help` 为准。

### 4.5 如何理解 `trtexec` 输出

- **Throughput**：单位时间完成的推理数量。
- **Latency**：单次推理从提交到完成的时间；要注意统计的是 Host、GPU 还是端到端路径。
- **GPU Compute Time**：GPU 执行网络计算的时间。
- **H2D / D2H**：Host-to-Device 和 Device-to-Host 数据传输时间。
- **Total Host Walltime**：从 Host 角度观察到的总耗时。
- **Enqueue Time**：Host 向 CUDA Stream 提交工作的时间。

如果 GPU Compute Time 很低但 Total Host Walltime 很高，瓶颈可能来自数据传输、Host 调度、同步或前后处理，而不是 TensorRT Kernel 本身。

## 5. TensorRT Python API：从 ONNX 到 Engine

### 5.1 构建对象的顺序

Python API 中的基本构建流程是：

```text
Logger
  -> Builder
  -> NetworkDefinition
  -> OnnxParser
  -> BuilderConfig
  -> build_serialized_network
  -> 保存 plan
```

### 5.2 解析 ONNX 并保存 Serialized Engine

```python
# ===== 使用 TensorRT Python API 构建 Engine =====
from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path: str, engine_path: str, use_fp16: bool = True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 当前 TensorRT 网络需要显式描述输入输出；STRONGLY_TYPED 表示类型由网络显式决定。
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Parser 失败时逐条打印错误，便于定位不支持的算子或形状问题。
    if not parser.parse_from_file(onnx_path):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError(f"failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    # Workspace 是 Builder 可使用的临时设备内存上限。
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if use_fp16:
        # 允许 Builder 为支持的层选择 FP16 实现。
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("failed to build TensorRT engine")

    Path(engine_path).write_bytes(bytes(serialized))
    return bytes(serialized)


if __name__ == "__main__":
    build_engine("tiny_classifier.onnx", "tiny_classifier.plan")
```

这里的 `build_serialized_network` 返回的是序列化结果。保存后，部署进程可以只使用 Runtime 读取 plan，而不必重新执行 Builder。

### 5.3 手动查看 Network I/O

```python
# ===== 在构建后读取网络输入输出元数据 =====
for index in range(network.num_inputs):
    tensor = network.get_input(index)
    print("input:", tensor.name, tensor.dtype, tensor.shape)

for index in range(network.num_outputs):
    tensor = network.get_output(index)
    print("output:", tensor.name, tensor.dtype, tensor.shape)
```

真实代码中应在构建前确认：

- 输入名称与导出时一致。
- 输入顺序、布局和 dtype 正确。
- 输出张量的语义和后处理逻辑匹配。
- 动态维度已经通过 Profile 覆盖。

## 6. TensorRT Python Runtime API

### 6.1 Runtime 的执行对象

加载 Engine 后，需要创建 `ExecutionContext`：

```python
# ===== 从 plan 文件创建 Runtime、Engine 和 Context =====
from pathlib import Path

import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine_bytes = Path("tiny_classifier.plan").read_bytes()
engine = runtime.deserialize_cuda_engine(engine_bytes)

if engine is None:
    raise RuntimeError("failed to deserialize engine")

# 一个 Engine 可以创建多个 Context，用于不同的执行状态或并发任务。
context = engine.create_execution_context()

for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    print(name, mode, shape, dtype)
```

### 6.2 Tensor 地址与异步执行

TensorRT Runtime 不替 Python 自动管理所有 GPU 缓冲区。应用需要准备输入和输出设备内存，并通过 `set_tensor_address` 把指针交给 Context。

下面使用 CUDA Python 的 Runtime API 表示内存分配和拷贝。示例强调执行顺序，实际程序也可以使用 PyTorch、CuPy 或 Numba 分配 GPU Tensor，并传入它们的设备地址。

```python
# ===== 使用 CUDA Python 设备指针执行一次 TensorRT 推理 =====
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def check_cuda(error):
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {error}")


def run_once(engine_path: str, input_array: np.ndarray):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_array = np.ascontiguousarray(input_array, dtype=np.float32)

    # 动态输入需要在执行前设置实际形状；静态输入也可以显式设置。
    context.set_input_shape(input_name, input_array.shape)
    output_shape = tuple(context.get_tensor_shape(output_name))
    output_array = np.empty(output_shape, dtype=np.float32)

    error, device_input = cudart.cudaMalloc(input_array.nbytes)
    check_cuda(error)
    error, device_output = cudart.cudaMalloc(output_array.nbytes)
    check_cuda(error)
    error, stream = cudart.cudaStreamCreate()
    check_cuda(error)

    try:
        # H2D：把连续的 Host 输入复制到 Device 缓冲区。
        check_cuda(cudart.cudaMemcpyAsync(
            device_input,
            input_array.ctypes.data,
            input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ))

        # 绑定每个 I/O Tensor 的设备地址。
        context.set_tensor_address(input_name, int(device_input))
        context.set_tensor_address(output_name, int(device_output))

        # 将推理工作提交到指定 CUDA Stream。
        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")

        # D2H：排队把输出复制回 Host；同步后才能读取 output_array。
        check_cuda(cudart.cudaMemcpyAsync(
            output_array.ctypes.data,
            device_output,
            output_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        ))
        check_cuda(cudart.cudaStreamSynchronize(stream))
        return output_array
    finally:
        # 释放设备内存和 CUDA Stream，避免重复调用时泄漏资源。
        cudart.cudaFree(device_input)
        cudart.cudaFree(device_output)
        cudart.cudaStreamDestroy(stream)


output = run_once("tiny_classifier.plan", np.random.rand(1, 3, 32, 32))
print(output.shape)
```

### 6.3 为什么要使用 CUDA Stream

**CUDA Stream** 是按顺序执行的一组 GPU 操作。把 H2D、TensorRT Kernel 和 D2H 放入同一个 Stream，可以表达数据依赖：

```text
H2D(input)
   -> TensorRT enqueue
   -> D2H(output)
   -> Stream synchronize
```

`execute_async_v3` 表示向 Stream 提交工作；它不是“调用后 GPU 已经完成”的同步接口。应用必须在读取输出前使用 Stream 同步、Event 或其他 CUDA 等待机制。

### 6.4 静态形状和动态形状的 Runtime 差异

| 场景 | 构建阶段 | 执行阶段 |
|---|---|---|
| 静态形状 | 输入维度全部确定 | 直接绑定地址并执行 |
| 动态形状 | 为动态输入创建 Profile | 先选择/使用覆盖形状，再设置实际输入 shape |
| 动态输出 | 构建时可能无法得到完整输出尺寸 | 设置输入后从 Context 查询输出尺寸 |

如果动态输出的尺寸依赖输入数据本身，应用需要在分配输出缓冲区之前查询 Context 的实际输出形状，并处理无法确定的输出维度。

## 7. 动态形状与 Optimization Profile

### 7.1 为什么需要 Profile

动态形状给模型带来灵活性，但 Builder 不能只凭一个未知形状选择最优 Kernel。Optimization Profile 为每个动态输入提供 `min / opt / max` 三个边界，使 Builder 可以在允许范围内进行选择和调优。

### 7.2 Python API 创建 Profile

```python
# ===== 为动态 batch 创建 Optimization Profile =====
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

if not parser.parse_from_file("tiny_classifier_dynamic.onnx"):
    raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
profile = builder.create_optimization_profile()

# 同一个动态输入必须给出完整的 min/opt/max 形状。
profile.set_shape(
    "images",
    min=(1, 3, 32, 32),
    opt=(4, 3, 32, 32),
    max=(8, 3, 32, 32),
)
config.add_optimization_profile(profile)

serialized = builder.build_serialized_network(network, config)
```

动态形状常见错误包括：没有添加 Profile、输入形状超出 Profile、多个输入的 Profile 不一致，以及设置输入 shape 后仍按旧输出尺寸分配缓冲区。

### 7.3 多 Profile

如果实际输入分布有明显不同的区域，可以创建多个 Profile，例如：

```text
Profile 0: batch 1 ~ 4
Profile 1: batch 8 ~ 32
```

多个 Profile 会增加构建时间、Engine 大小和资源规划复杂度。只有当输入范围确实需要不同的优化策略时，才增加 Profile。

## 8. 精度与量化

### 8.1 FP32、TF32、FP16、BF16 和 INT8

| 格式 | 基本特点 | 典型取舍 |
|---|---|---|
| FP32 | 单精度浮点 | 数值范围和精度较稳，显存和算力开销较高 |
| TF32 | 使用 FP32 范围和较短尾数的 Tensor Core 格式 | 适合部分 FP32 矩阵计算，通常由硬件支持 |
| FP16 | 半精度浮点 | 速度和显存通常更有优势，但范围和精度更有限 |
| BF16 | 指数范围接近 FP32、尾数短 | 比 FP16 更不容易溢出，但精度特性不同 |
| INT8 | 8 位整数 | 内存和计算效率高，需要量化尺度或校准 |

Builder 的精度标志表示允许使用某种实现，不代表所有层强制使用该格式。实际选择还受 GPU、层类型、输入输出约束和准确性要求影响。

### 8.2 FP16 构建

```bash
trtexec \
  --onnx=tiny_classifier.onnx \
  --saveEngine=tiny_classifier_fp16.plan \
  --fp16
```

验证 FP16 时需要比较：

1. 输出数值与 FP32 基线的误差。
2. 任务指标是否变化。
3. 单次延迟、吞吐和显存。
4. 是否出现 `Inf` 或 `NaN`。

Softmax、Sigmoid、长序列归约和数值范围很大的中间结果可能对精度更敏感。必要时可以让关键计算保留更高精度，形成混合精度网络。

### 8.3 INT8 的基本概念

**量化（quantization）**把浮点值映射到有限的整数表示。**PTQ（Post-Training Quantization）**在训练完成后量化；**QAT（Quantization-Aware Training）**在训练过程中模拟量化误差，让模型适应量化。

INT8 通常需要知道激活和权重的 scale：

```text
float value -- scale / rounding --> int8 value
int8 value -- dequantize --------> float approximation
```

校准（calibration）使用有代表性的输入数据估计激活范围。校准数据不需要包含所有训练数据，但应覆盖实际推理中的输入分布。

### 8.4 INT8 校准的注意点

- 校准输入的预处理必须与真实推理一致。
- 校准数据数量太少或分布偏移会导致 scale 不可靠。
- 动态形状网络需要为校准指定合适的 Calibration Profile。
- 量化后必须重新检查准确率、溢出、饱和和异常值。
- 某些层适合保持 FP16/FP32，不应机械地将整个网络强制 INT8。

对于明确包含 `QuantizeLinear` / `DequantizeLinear` 的 ONNX 图，量化信息由图中的 Q/DQ 节点表达；这与依靠校准器在构建阶段估计范围的流程不同。

## 9. Builder 优化与运行时性能

### 9.1 Layer Fusion

**Layer Fusion** 是把多个连续操作合并成更少的执行单元，例如把卷积、偏置和激活融合。融合可以减少 Kernel 启动次数、中间 Tensor 写回和内存访问，但是否能融合取决于图结构、数据类型和形状。

### 9.2 Tactic

**Tactic** 是某个层或算子的一种具体实现方式。不同 Tactic 可能使用不同的 CUDA Kernel、Tile 形状、访存方式或库调用。

Builder 会在构建时测量候选 Tactic，并选择符合约束且性能较好的方案。因此：

- 构建时间可能明显长于加载时间。
- 目标 GPU 和输入形状会影响选择结果。
- 重新构建可能因为时钟、环境或可用资源不同而产生不同结果。
- 不要只比较 Engine 文件大小，要用同一套基准重新测量。

### 9.3 Workspace 和内存池

Workspace 是 Builder 或运行时算法可以使用的临时设备内存。较大的上限可能让 Builder 找到更多 Tactic，但也会增加构建或运行时的资源压力。

```python
# ===== 设置 Workspace 内存池上限 =====
config = builder.create_builder_config()
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    1 << 30,  # 允许使用 1 GiB 临时空间
)
```

Workspace 上限不是“模型一定会占用的显存”，而是允许相关算法使用的内存范围。实际占用还包括权重、激活、I/O 缓冲区和 CUDA Runtime 资源。

### 9.4 Timing Cache

Timing Cache 保存 Builder 对候选实现的测量结果。重复构建相同或相近网络时，可以复用部分计时信息，减少构建时间。

Timing Cache 不是推理结果缓存，也不是 Engine。它不能替代 Engine 序列化，并且在硬件、TensorRT 版本或构建条件变化时需要谨慎复用。

### 9.5 CUDA Graph

当单次推理计算很快、Host 端反复提交 Kernel 的开销占比较高时，可以考虑 CUDA Graph。它把一组稳定的 CUDA 操作捕获成可重复提交的执行图。

CUDA Graph 通常要求输入形状、内存地址和执行路径相对稳定。动态形状、数据依赖输出、同步插件或变化的 I/O 缓冲区可能限制使用条件。

## 10. 性能测量

### 10.1 Latency、Throughput 和端到端时间

- **Latency**：一次推理从输入可用到输出可用所经历的时间。
- **Throughput**：单位时间完成的推理数量。
- **GPU time**：GPU 执行 Kernel 的时间。
- **End-to-end latency**：包含预处理、H2D、TensorRT 执行、D2H 和后处理的完整耗时。

只报告 GPU Kernel 时间不能代表完整应用体验；只报告端到端时间又可能隐藏模型本身的瓶颈。测量时应明确计时范围。

### 10.2 稳定基准的基本步骤

```text
固定模型、输入形状和精度
  -> 预热
  -> 重复多次执行
  -> 使用 CUDA Event 或 trtexec 统计
  -> 查看平均值、P50、P95/P99
  -> 记录 GPU、驱动、CUDA、TensorRT 和功耗温度条件
```

预热用于排除首次加载、内存分配、Kernel 初始化和缓存建立等一次性成本。平均延迟不能代替尾延迟；交互式或实时任务通常还需要观察 P95/P99。

### 10.3 用 `trtexec` 做可复现测量

```bash
# 固定预热时间和最短测试时长，并保留数据传输在统计中
trtexec \
  --loadEngine=tiny_classifier_fp16.plan \
  --shapes=images:1x3x32x32 \
  --warmUp=500 \
  --duration=10 \
  --percentile=50,95,99

# 只看网络在 GPU 上的执行，排除 H2D/D2H
trtexec \
  --loadEngine=tiny_classifier_fp16.plan \
  --shapes=images:1x3x32x32 \
  --noDataTransfers \
  --warmUp=500 \
  --duration=10
```

同一组对比实验应固定输入 shape、batch、精度、线程和 GPU 状态。GPU 时钟、温度、功耗限制和数据传输路径都可能改变结果。

### 10.4 分层分析

```bash
# 查看网络层级信息
trtexec \
  --loadEngine=tiny_classifier_fp16.plan \
  --dumpLayerInfo \
  --profilingVerbosity=detailed

# 使用 Nsight Systems 查看 CPU、CUDA Stream 和 GPU 时间线
nsys profile -o trt_timeline trtexec \
  --loadEngine=tiny_classifier_fp16.plan \
  --iterations=50 \
  --warmUp=0 \
  --duration=0
```

Nsight Systems 适合观察完整时间线和同步关系；TensorRT 的层级信息或 Nsight Compute 更适合定位单个 Kernel、访存、Occupancy 和指令瓶颈。

## 11. 正确性与精度验证

### 11.1 为什么不能只看速度

一个 Engine 即使延迟降低，也可能因为预处理不一致、输出布局误解、精度下降或后处理错误而产生错误结果。速度和正确性必须一起验证。

### 11.2 输出误差的简单计算

```python
# ===== 比较 TensorRT 输出和参考输出 =====
import numpy as np


def compare_outputs(reference: np.ndarray, actual: np.ndarray):
    absolute_error = np.abs(reference - actual)
    relative_error = absolute_error / np.maximum(np.abs(reference), 1e-6)

    print("max absolute error:", absolute_error.max())
    print("mean absolute error:", absolute_error.mean())
    print("max relative error:", relative_error.max())


reference = np.array([0.2, 0.5, 0.3], dtype=np.float32)
actual = np.array([0.2001, 0.4999, 0.3], dtype=np.float32)
compare_outputs(reference, actual)
```

### 11.3 参考结果的选择

可以按以下顺序建立基线：

```text
训练框架 FP32
  -> ONNX Runtime
  -> TensorRT FP32
  -> TensorRT FP16/BF16
  -> TensorRT INT8
```

每次只改变一个主要因素，并同时记录数值误差和任务指标。对于分类，应比较 Top-1/Top-k 或混淆矩阵；对于回归，应比较 MAE/MSE；对于检测和分割，应使用对应任务指标。

## 12. Plugin 与不支持算子

### 12.1 什么时候需要 Plugin

如果 ONNX 图包含 TensorRT 原生层不支持的算子，或者需要使用自定义 CUDA Kernel，就可能需要 Plugin。Plugin 是 TensorRT 网络中的自定义层，实现了输入输出描述、配置、序列化以及执行逻辑。

### 12.2 处理不支持算子的顺序

```text
确认 Parser 报错节点
  -> 检查 TensorRT Operator 支持情况
  -> 尝试改写模型图或使用等价算子
  -> 使用 ONNX GraphSurgeon 等工具处理图
  -> 最后编写或引入 Plugin
```

不要一看到 Parser 报错就立即写 Plugin。先确认问题是算子不支持、输入 shape 不明确、dtype 不匹配，还是 ONNX 导出图本身存在错误。

### 12.3 Plugin 需要解决的问题

- 输入和输出 Tensor 的形状、dtype 和格式。
- 静态和动态 shape 的配置逻辑。
- Workspace 大小和临时内存。
- CUDA Stream 上的异步执行。
- Plugin 字段、版本和序列化数据。
- 不同 GPU 架构与精度模式。
- 数值正确性、边界条件和错误报告。

## 13. Engine 兼容性与安全

### 13.1 Engine 的兼容性边界

Engine 可能依赖以下条件：

- TensorRT 版本。
- CUDA、Driver 和操作系统。
- GPU 架构或 Compute Capability。
- 构建时启用的精度、Plugin 和内存配置。
- 输入输出名称、shape、dtype 和 Profile。

跨环境部署时，不要只复制一个 `.plan` 文件。应同时保存构建命令、模型来源、版本信息和验证结果；必要时在目标设备重新构建。

### 13.2 Engine 文件的安全性

Engine 是包含可执行 CUDA Tactic 的二进制部署产物。不要反序列化来源不明或不可信的 Engine 文件。部署程序应限制 Engine 来源，并在加载前校验文件路径、权限和版本信息。

## 14. 常见问题定位

### 14.1 Parser 失败

可能原因：

- ONNX 中存在 TensorRT 不支持的算子。
- opset 或算子属性不符合当前 Parser 支持范围。
- shape 推导失败或动态维度缺少约束。
- dtype、布局或输入输出名称不符合预期。
- 导出模型仍包含训练态或框架特有逻辑。

定位顺序：保存完整 Parser 日志、记录失败节点、用 ONNX 工具确认节点输入输出，再决定改图还是使用 Plugin。

### 14.2 Engine 构建失败

可能原因：

- Workspace 或其他内存池上限不足。
- Profile 未覆盖动态输入。
- GPU 不支持请求的精度或算子。
- 某个 Tactic 需要的资源超出设备限制。
- Plugin 没有正确注册或序列化。

可以先使用更简单的精度和静态 shape 验证网络，再逐步加入动态 shape、FP16、INT8 和 Plugin。

### 14.3 Runtime 执行失败

优先检查：

1. 输入输出 Tensor 名称是否与 Engine 一致。
2. 每个 I/O Tensor 是否都绑定了有效设备地址。
3. 动态输入是否调用了 `set_input_shape`。
4. 输出 shape 是否在分配缓冲区前重新查询。
5. dtype、连续性和布局是否匹配。
6. `execute_async_v3` 使用的 Stream 是否有效。
7. H2D、执行、D2H 和同步顺序是否正确。

### 14.4 速度没有提升

TensorRT 只优化网络执行段，端到端速度还可能受以下因素影响：

- Host 预处理或后处理。
- H2D/D2H 传输。
- Python 调度和内存分配。
- Batch 太小导致 GPU 利用率不足。
- Kernel 启动和同步开销。
- GPU 时钟、温度、功耗和频率变化。

应先用 `trtexec` 得到网络基线，再在真实应用中分别测量预处理、传输、TensorRT 执行和后处理。

## 15. TensorRT 生态中的相关工具

### 15.1 TensorRT 与 ONNX Runtime

ONNX Runtime 是通用推理运行时，可以选择 TensorRT Execution Provider。直接使用 TensorRT Python/C++ API 时，可以获得更直接的 Engine 构建和执行控制；使用 ONNX Runtime 时，应用接口更统一，但需要理解 Provider 的配置和 Engine 缓存行为。

### 15.2 TensorRT 与 Torch-TensorRT

Torch-TensorRT 面向 PyTorch 模型提供编译和集成路径，适合希望保留 PyTorch 调用方式的场景。直接 TensorRT API 更接近 Engine 和 Runtime 本身，适合学习构建过程、管理 I/O 和集成到非 PyTorch 应用。

### 15.3 TensorRT 与 TensorRT-LLM

TensorRT 是通用推理优化 SDK；TensorRT-LLM 是面向大语言模型的专用推理框架，额外关注 KV Cache、Attention、并行和批处理调度。二者有关联，但 TensorRT 的基础 Engine/Runtime 概念不等于 TensorRT-LLM 的完整生成服务栈。

### 15.4 TensorRT 与 Triton Inference Server

Triton 是模型服务系统，负责模型仓库、请求调度、批处理、协议和服务监控；TensorRT 负责模型图优化和 GPU 执行。Triton 可以把 TensorRT Engine 作为一种后端模型格式加载。

## 16. 一个通用的学习实验

可以用一个小型分类模型完整走一遍以下流程：

```text
定义 PyTorch 模型
  -> 导出静态 ONNX
  -> ONNX checker 验证
  -> trtexec 构建 FP32 Engine
  -> trtexec 构建 FP16 Engine
  -> Python Runtime 执行
  -> 对比 PyTorch/TensorRT 输出
  -> 测量 latency、throughput 和显存
```

建议保存以下记录：

| 类别 | 记录内容 |
|---|---|
| 模型 | ONNX 文件、输入输出名称、opset、参数量 |
| 环境 | GPU、Driver、CUDA、TensorRT、操作系统 |
| 构建 | `trtexec` 命令、精度、Profile、Workspace |
| 正确性 | 参考输出、最大误差、任务指标 |
| 性能 | Warmup、迭代次数、平均延迟、P50/P95/P99、吞吐 |
| 资源 | 显存、GPU 利用率、温度、功耗和时钟 |

## 17. 学习顺序

1. 理解训练模型、ONNX、Engine、Runtime 和 ExecutionContext 的关系。
2. 安装 TensorRT，确认 `trtexec --help`、Python import 和 GPU 可用。
3. 用静态 shape ONNX 跑通 `trtexec` 构建与测量。
4. 用 Python API 解析 ONNX、保存 Engine，并读取 I/O 元数据。
5. 用 Runtime API 完成设备内存、地址绑定、Stream 和 `execute_async_v3`。
6. 学习动态 shape、Profile 和动态输出缓冲区管理。
7. 比较 FP32、FP16、BF16 和 INT8 的性能与准确性。
8. 学习 Layer Fusion、Tactic、Workspace、Timing Cache 和 CUDA Graph。
9. 使用 Parser 日志、Engine Inspector、Nsight Systems 和 Compute Sanitizer 定位问题。
10. 最后再学习 Plugin、TensorRT-LLM、Triton 和更复杂的服务集成。

## 18. 参考资料

- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [Installing TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [Python API Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)
- [Inference Library Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/index.html)
- [Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)
- [Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [Performance Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
- [Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
