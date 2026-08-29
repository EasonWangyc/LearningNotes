# ONNX — Open Neural Network Exchange

> 模型交换格式与跨框架推理学习笔记

## 内容概览

- ONNX 格式与计算图
- PyTorch / TensorFlow 模型导出为 ONNX
- ONNX Runtime 推理
- 模型优化与量化

## 1. ONNX 是什么？

**ONNX（Open Neural Network Exchange）** 是一种开放的模型格式，作为不同框架之间的**中间表示（IR）**。

```
PyTorch ──┐
TensorFlow─┤──► ONNX 格式 ──► ONNX Runtime / TensorRT / OpenVINO / ...
Scikit ───┘                    （跨平台高效推理）
```

### ONNX 计算图结构

| 元素 | 说明 |
|------|------|
| **Node** | 算子节点（Conv、MatMul、ReLU...） |
| **Initializer** | 模型权重（常量张量） |
| **Graph** | 有向无环图（DAG），描述数据流 |
| **Input/Output** | 模型的输入输出张量规格 |

**查看 ONNX 模型**：使用 [Netron](https://netron.app/) 可视化网络结构

---

## 2. PyTorch → ONNX 导出原理

PyTorch 使用 **tracing**（追踪）或 **scripting** 方式导出：

- **torch.onnx.export**：追踪一次前向传播，记录所有算子
- 注意：动态控制流（if/for 依赖输入值）无法被 trace，需用 `torch.jit.script`

### 动态 Batch Size

导出时设置 `dynamic_axes` 让某些维度可变：
```python
dynamic_axes = {
    'input': {0: 'batch_size'},   # 第0维（batch）动态
    'output': {0: 'batch_size'},
}
```

---

## 3. ONNX Runtime 推理

ONNX Runtime（ORT）支持多种执行提供者（Execution Provider）：

| Provider | 说明 |
|----------|------|
| `CPUExecutionProvider` | 默认，跨平台 |
| `CUDAExecutionProvider` | NVIDIA GPU 加速 |
| `TensorrtExecutionProvider` | TensorRT 后端 |
| `OpenVINOExecutionProvider` | Intel 硬件优化 |

---

## 4. 模型优化与量化

### 图优化（Graph Optimization）
ORT 自动执行：
- **常量折叠**：编译期计算常量表达式
- **算子融合**：Conv + BN + ReLU → 单个融合算子
- **冗余节点消除**

### 量化（Quantization）
将 FP32 权重/激活量化为 INT8，速度提升 2-4x：

| 方式 | 说明 |
|------|------|
| **PTQ（训练后量化）** | 无需重训练，需少量校准数据 |
| **QAT（量化感知训练）** | 训练中模拟量化误差，精度更高 |
| **动态量化** | 推理时动态量化激活，适合 NLP |

---

## 5. ONNX 在部署链路中的位置

ONNX 最重要的价值不是“推理框架”，而是**模型中间表示**。真实部署时常见链路如下：

```text
训练框架
PyTorch / TensorFlow / Paddle
        |
        | 导出
        v
ONNX 模型
        |
        | 检查、简化、shape inference、精度对齐
        v
推理后端
ONNX Runtime / TensorRT / OpenVINO / RKNN / NCNN
        |
        v
业务部署
服务端 GPU / 边缘 NPU / CPU / 移动端
```

面试回答可以抓住三句话：

1. ONNX 是跨框架的 IR，描述的是计算图、权重、输入输出和算子版本。
2. ONNX 本身不保证高性能，性能取决于后端执行器和硬件。
3. 导出后必须做结构检查、数值对齐和目标后端兼容性验证。

## 6. ONNX 文件里到底有什么

| 组件 | 作用 | 常见排查点 |
|---|---|---|
| `ModelProto` | 整个模型，包含 IR version、opset、graph、metadata | opset 是否被目标后端支持 |
| `GraphProto` | 计算图，包含 node、input、output、initializer | 输入输出名、拓扑结构是否正确 |
| `NodeProto` | 单个算子节点，例如 Conv、Relu、MatMul | 算子属性、domain、是否有自定义算子 |
| `TensorProto` | 权重或常量张量 | 大模型是否 external data |
| `ValueInfoProto` | tensor 的类型和 shape 信息 | 动态维度、缺失 shape |
| `opset_import` | 算子集版本 | 版本太新/太旧都会导致后端不支持 |

### IR version 和 opset 的区别

| 概念 | 含义 |
|---|---|
| IR version | ONNX 文件结构本身的版本，类似“容器格式版本” |
| opset version | 算子语义版本，决定 Conv、Resize、TopK 等算子的定义 |

工程中更常遇到的是 **opset 兼容性问题**。例如某个后端只支持到 opset 17，而模型导出成 opset 19，就可能出现解析失败或算子行为不一致。

## 7. 导出 ONNX 的工程要点

### 7.1 导出前先固定模型状态

```python
model.eval()
with torch.no_grad():
    torch.onnx.export(...)
```

原因：

- `eval()` 会固定 BatchNorm、Dropout 等训练/推理行为。
- `no_grad()` 避免无意义的梯度图和额外开销。
- dummy input 的 shape、dtype、device 应尽量贴近部署输入。

### 7.2 动态 shape 不要滥用

动态 batch 很常见，但动态 H/W 要谨慎：

```python
dynamic_axes = {
    "images": {0: "batch"},
    "logits": {0: "batch"},
}
```

| 选择 | 优点 | 缺点 |
|---|---|---|
| 固定 shape | 后端优化最充分，TensorRT/RKNN 更容易编译 | 灵活性差 |
| 动态 batch | 兼顾吞吐和灵活性 | 需要 profile 或额外调优 |
| 动态 H/W | 输入尺寸灵活 | shape inference、算子支持、engine 优化更复杂 |

部署时优先用**固定 H/W + 动态 batch**。如果业务必须动态分辨率，建议先确认目标后端是否支持，并准备典型尺寸做 profile。

### 7.3 导出后必须做三类检查

```python
import onnx
from onnx import checker, shape_inference

model = onnx.load("model.onnx")
checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.save(model, "model_infer_shape.onnx")
```

| 检查 | 工具 | 目的 |
|---|---|---|
| 结构合法性 | `onnx.checker.check_model` | 检查 protobuf、opset、图结构是否合法 |
| shape 推断 | `onnx.shape_inference.infer_shapes` | 补充中间 tensor shape，方便排查和量化 |
| 可视化 | Netron | 看输入输出名、数据布局、异常节点 |
| 数值对齐 | ONNX Runtime vs 原框架 | 确认导出没有改变模型语义 |

注意：ONNX shape inference 不是万能的。遇到动态 `Reshape`、复杂控制流、自定义算子时，中间 shape 可能无法完全推断。

## 8. 常见导出问题

| 问题 | 现象 | 处理 |
|---|---|---|
| Python 控制流被 trace 固化 | 输入变化后分支不对 | 使用 `torch.jit.script` 或新版 `torch.export` / dynamo 导出 |
| 自定义 op | ORT/TensorRT/RKNN 解析失败 | 改写为标准算子，或写 custom op/plugin |
| `Resize` 行为不一致 | 检测/分割结果偏移 | 检查 opset、`align_corners`、`coordinate_transformation_mode` |
| 动态 shape 太多 | TensorRT profile 难配，RKNN 转换失败 | 固定 H/W，只保留动态 batch |
| 输入 layout 混乱 | 输出精度明显不对 | 明确 NCHW/NHWC、RGB/BGR、归一化 |
| 大模型超过 2GB | 保存或 checker 失败 | 使用 ONNX external data |

## 9. ONNX Runtime 重点

### 9.1 Execution Provider

ONNX Runtime 会把图分配给不同 Execution Provider。常见策略：

```python
providers = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

顺序很重要：越靠前优先级越高。如果某些节点后端不支持，可能 fallback 到后面的 provider。排查性能时要确认模型是否真的跑在目标后端上。

### 9.2 图优化等级

ONNX Runtime 图优化分为 Basic、Extended、Layout 等层级，默认通常会开启优化。工程中常见用法：

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "optimized.onnx"
session = ort.InferenceSession("model.onnx", sess_options)
```

| 优化 | 例子 | 说明 |
|---|---|---|
| Basic | 常量折叠、Identity 删除、Conv+BN 融合 | 通常安全 |
| Extended | Attention/GELU/LayerNorm 融合 | 与 EP 有关 |
| Layout | NCHWc 等布局优化 | 可能和硬件相关 |

离线保存 optimized model 可以减少服务启动时优化开销，但要注意：离线优化模型通常和 provider、硬件、ORT 版本相关，不要随意跨环境复用。

### 9.3 I/O Binding

GPU 推理慢不一定是算子慢，可能是 H2D/D2H 拷贝慢。默认 `session.run()` 常把输入从 CPU 拷到 GPU，再把输出从 GPU 拷回 CPU。高性能部署可使用 I/O Binding，把输入输出绑定到设备内存，减少不必要拷贝。

适用场景：

- 前后处理也在 GPU 上完成。
- 模型输出会继续给下一个 GPU 模型。
- batch 大、输出大，数据拷贝占比明显。

## 10. ONNX 量化补充

ONNX Runtime 的 INT8 量化主要有两种表示：

| 格式 | 特点 | 常见场景 |
|---|---|---|
| QOperator | 用 `QLinearConv`、`MatMulInteger` 等量化算子替换原算子 | 传统 ORT INT8 |
| QDQ | 在原图中插入 `QuantizeLinear/DeQuantizeLinear` | QAT、TensorRT 显式量化更常见 |

动态量化和静态量化的区别：

| 方法 | 激活 scale/zero_point | 适合模型 |
|---|---|---|
| 动态量化 | 推理时动态计算 | Transformer、RNN、Linear 多的模型 |
| 静态量化 | 校准集提前统计并写入模型 | CNN、视觉模型、边缘部署 |

静态量化基本流程：

```text
FP32 ONNX
  -> shape inference / preprocess
  -> calibration data reader
  -> quantize_static
  -> 精度对齐
  -> 部署
```

精度下降排查顺序：

1. 先确认 FP32 ONNX 和原框架输出一致。
2. 再比较 INT8 ONNX 和 FP32 ONNX 输出。
3. 检查校准集是否代表真实数据分布。
4. 尝试 per-channel、不同 calibration method、排除敏感节点。
5. 若 PTQ 不满足，再考虑 QAT。

## 11. 部署排查 Checklist

| 阶段 | 必查项 |
|---|---|
| 导出前 | `model.eval()`、dummy input shape、dtype、前处理 |
| 导出时 | opset、input/output name、dynamic axes |
| 导出后 | `onnx.checker`、shape inference、Netron |
| 数值对齐 | 原框架 vs FP32 ONNX，误差阈值 |
| 后端转换 | TensorRT/RKNN/OpenVINO 是否支持所有 op |
| 性能测试 | 是否包含前后处理、H2D/D2H、warmup、batch |
| 生产部署 | engine/model 是否绑定硬件、版本、driver、CUDA |

## 12. 面试速记

**Q：ONNX 是什么？**

ONNX 是一种开放模型中间表示，用 protobuf 描述计算图、权重、输入输出和算子版本。它解决的是训练框架和推理后端之间的模型交换问题。

**Q：ONNX 导出后为什么还要校验？**

因为导出可能固化控制流、改变算子语义、丢失动态 shape 信息，目标后端也可能不支持某些 opset 或自定义算子。必须用 checker、shape inference、Netron 和数值对齐确认模型可用。

**Q：opset 太高是不是一定好？**

不一定。新 opset 表达能力更强，但目标后端可能不支持。部署时应选“训练框架能稳定导出、目标后端明确支持”的 opset。

**Q：ONNX Runtime 比 PyTorch 快在哪里？**

ORT 会做图优化、算子融合、provider 分配，并使用 CPU/CUDA/TensorRT/OpenVINO 等后端执行。速度提升来自图级优化和更适合推理的 runtime，而不是 ONNX 文件本身。

参考资料：

- ONNX Shape Inference：<https://onnx.ai/onnx/repo-docs/ShapeInference.html>
- ONNX External Data：<https://onnx.ai/onnx/repo-docs/ExternalData.html>
- ONNX Runtime Graph Optimizations：<https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html>
- ONNX Runtime Quantization：<https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- ONNX Runtime I/O Binding：<https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html>

```python
import torch
import torch.nn as nn
import numpy as np

# ===== 1. 定义一个简单模型 =====
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

model = SimpleModel().eval()
dummy_input = torch.randn(1, 3, 32, 32)
print(f"模型输出形状: {model(dummy_input).shape}")

# ===== 2. 导出为 ONNX =====
import os
onnx_path = "simple_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,                      # 推荐使用较新的 opset
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},        # batch 维度动态
        "output": {0: "batch_size"},
    },
    do_constant_folding=True,              # 常量折叠优化
)
print(f"\nONNX 模型已导出到: {onnx_path}")
print(f"文件大小: {os.path.getsize(onnx_path) / 1024:.1f} KB")

# ===== 3. 验证 ONNX 模型 =====
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("\nONNX 模型结构验证通过！")

    # 打印输入输出信息
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name}, shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}, shape={shape}")

    # 打印算子统计
    op_types = [node.op_type for node in onnx_model.graph.node]
    from collections import Counter
    print(f"\n算子分布: {dict(Counter(op_types))}")
except ImportError:
    print("安装 onnx: pip install onnx")

# ===== 4. ONNX Runtime 推理 =====
try:
    import onnxruntime as ort

    # 创建推理会话（自动选择最优 Provider）
    providers = ort.get_available_providers()
    print(f"\n可用 Providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先 CUDA
    )

    # 推理
    input_name = session.get_inputs()[0].name
    np_input = np.random.randn(4, 3, 32, 32).astype(np.float32)  # batch=4（动态）
    outputs = session.run(None, {input_name: np_input})
    print(f"ORT 推理输出形状: {outputs[0].shape}")

    # 与 PyTorch 结果对比
    with torch.no_grad():
        pt_out = model(torch.from_numpy(np_input)).numpy()
    print(f"最大差异 (PyTorch vs ONNX Runtime): {np.max(np.abs(pt_out - outputs[0])):.2e}")

except ImportError:
    print("安装 onnxruntime: pip install onnxruntime  (GPU版: onnxruntime-gpu)")

# ===== 5. ONNX 模型量化（PTQ）=====
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_path = "simple_model_int8.onnx"
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    orig_size = os.path.getsize(onnx_path) / 1024
    quant_size = os.path.getsize(quantized_path) / 1024
    print(f"\n量化完成：FP32={orig_size:.1f}KB → INT8={quant_size:.1f}KB（压缩比 {orig_size/quant_size:.1f}x）")
except (ImportError, Exception) as e:
    print(f"\n量化跳过: {e}")

# 清理测试文件
for f in [onnx_path, "simple_model_int8.onnx"]:
    if os.path.exists(f):
        os.remove(f)
```

# RKNN学习

在使用瑞芯微系列开发板时遇到需要将onnx模型转换为rknn模型的需求，学习了相关的知识，记录如下。

## 1. RKNN 工具链速记

RKNN 是 Rockchip NPU 使用的模型格式。实际部署不是把 ONNX 直接丢到开发板上跑，而是先在 PC 端用 RKNN-Toolkit2 完成模型解析、图优化、量化和导出，再把 `.rknn` 放到开发板上，用 RKNN Runtime C/C++ API 或 RKNN-Toolkit-Lite2 Python API 调用 NPU 推理。

```text
PyTorch / TensorFlow / ONNX
        |
        |  PC 端 RKNN-Toolkit2
        v
load_onnx -> build(图优化 + 量化) -> export_rknn
        |
        v
开发板 Runtime / Lite2 -> RKNPU 驱动 -> NPU 推理
```

面试可以这样概括：RKNN-Toolkit2 负责转换、仿真、精度分析和性能评估；Runtime/Lite2 负责板端推理；RKNPU 驱动负责和 NPU 硬件交互。

## 2. ONNX 转 RKNN 基本流程

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
)
rknn.load_onnx('model.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('model.rknn')
rknn.release()
```

| 步骤 | 作用 | 面试重点 |
|---|---|---|
| `config` | 设置输入归一化、目标芯片、量化策略 | `mean_values/std_values` 要和训练、ONNX 推理前处理一致 |
| `load_onnx` | 解析 ONNX 图和算子 | opset、动态 shape、特殊算子可能导致转换失败 |
| `build` | 图优化、算子融合、量化、生成 NPU 图 | `do_quantization=True` 时需要校准数据 |
| `export_rknn` | 保存 `.rknn` | 板端部署使用 `.rknn`，不是直接用 ONNX |
| `init_runtime/inference` | 仿真或连板推理 | 用于对齐 ONNX、模拟器和板端输出 |

## 3. FP32 如何转换为 INT8

常见 INT8 后训练量化 PTQ 的核心是把浮点数映射到 8 bit 整数。公式可以记成：

```text
q = round(x / scale + zero_point)
x ≈ scale * (q - zero_point)
```

| 参数 | 含义 |
|---|---|
| `x` | 原始 FP32 值 |
| `q` | 量化后的 INT8/UINT8 整数 |
| `scale` | 浮点范围到整数范围的缩放比例 |
| `zero_point` | 浮点 0 对应的整数位置，非对称量化常用 |

直观理解：如果某层激活大致落在 `[min_val, max_val]`，工具会根据这个范围计算 `scale` 和 `zero_point`，把连续的 FP32 值压到有限的 8 bit 整数范围。NPU 推理时主要做整数计算，因此模型体积、内存带宽和计算开销都会下降，但会引入量化误差。

## 4. 为什么量化需要数据

| 对象 | 如何得到量化范围 | 是否需要校准数据 |
|---|---|---|
| 权重 weights | 模型文件里已经有权重，转换时可直接统计 | 通常不需要 |
| 激活 activations | 中间 tensor 的范围取决于输入样本 | 需要 |

`dataset.txt` 里的数据不是用来重新训练模型的，而是用来做校准 calibration。PTQ 不更新权重，只用代表性输入跑前向，统计每层激活范围。校准数据最好接近真实部署场景，例如摄像头角度、光照、目标大小、背景、类别分布都要尽量一致。常见做法是准备几十到几百张代表性图片，一行一个路径：

```text
./calib/0001.jpg
./calib/0002.jpg
./calib/0003.jpg
```

注意：校准前处理必须和真实推理一致，包括 resize、letterbox、RGB/BGR、NCHW/NHWC、归一化方式等。很多“模型转完精度不对”的问题，本质是前处理不一致。

## 5. PTQ、QAT、混合量化

| 方法 | 含义 | 优点 | 缺点 |
|---|---|---|---|
| FP16/FP32 | 不做 INT8，或只降到半精度 | 精度损失小 | 速度和内存收益较小 |
| PTQ | 训练后量化，直接用 FP32 模型和校准数据转换 | 成本低，工程最常见 | 对校准数据和异常值敏感 |
| QAT | 训练时插入伪量化，让模型适应量化误差 | 精度通常更稳 | 需要重新训练或微调 |
| 混合量化 | 敏感层保留浮点，其他层 INT8 | 折中精度和性能 | 需要分析和调参 |

RKNN-Toolkit2 常见相关参数：`do_quantization=True` 开启量化，`dataset='dataset.txt'` 指定校准集，`quantized_dtype='asymmetric_quantized-8'` 是 Toolkit2 常见 8 bit 非对称量化类型，`quantized_algorithm='normal'` 或 `'mmse'` 控制量化参数搜索方式，`quantized_method='channel'` 常用于按通道量化。

## 6. 转换后如何排查

1. 先用原始 ONNX 在 PC 上跑一批样本，保存输出或精度指标。
2. 用 RKNN-Toolkit2 模拟器跑同样输入，对比 ONNX 输出。
3. 如果 INT8 精度明显下降，优先检查前处理、layout、RGB/BGR 和校准集分布。
4. 如果少数层误差大，可尝试 `accuracy_analysis`、`mmse`、混合量化或 QAT。
5. 最后在开发板上测真实业务指标和 `eval_perf`，不要只看单次耗时。

| 现象 | 常见原因 |
|---|---|
| 转换失败 | ONNX opset 不支持、动态 shape、特殊算子、输入输出名不匹配 |
| 板端和 PC 结果不一致 | 前处理不一致、NCHW/NHWC 混淆、RGB/BGR 搞反 |
| INT8 精度掉很多 | 校准集不代表真实场景、异常值影响 scale、敏感层不适合量化 |
| 速度没有预期快 | 部分算子落到 CPU、模型结构不适合 NPU、内存拷贝开销大 |

## 7. 面试回答模板

**Q：你怎么理解 RKNN？**

RKNN 是 Rockchip NPU 的部署模型格式。工程上通常从 PyTorch 或 TensorFlow 导出 ONNX，再用 RKNN-Toolkit2 在 PC 端做模型加载、图优化、量化和导出，最后把 `.rknn` 放到开发板，通过 Runtime C API 或 Lite2 Python API 调用 NPU 推理。

**Q：FP32 怎么变成 INT8？**

本质是量化映射。工具根据权重和校准数据统计到的激活范围，计算每层或每通道的 `scale` 和 `zero_point`，用 `q = round(x / scale + zero_point)` 把浮点值映射成 8 bit 整数。这样推理更快、内存更省，但会有量化误差。

**Q：量化数据是不是训练数据？**

不是。PTQ 的校准数据不更新权重，只用于统计激活范围。它应该和真实部署输入分布接近，否则 INT8 的 scale 估计不准，精度可能下降。

**Q：INT8 精度下降怎么办？**

先检查前处理一致性，再换更有代表性的校准集；如果还不够，可以尝试 `mmse`、混合量化，把敏感层保留为浮点，必要时再做 QAT。

## 8. 一句话总结

RKNN 部署不是简单“ONNX 改后缀”，而是面向 Rockchip NPU 的编译和量化过程；面试重点通常是转换流程、INT8 量化原理、校准数据作用、精度下降排查和板端部署接口。

参考资料：

- RKNN-Toolkit2 官方仓库：<https://github.com/airockchip/rknn-toolkit2>
- RKNN-Toolkit2 API 差异文档：<https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_API_Difference_With_Toolkit1-2.3.2.md>
- RKNN YOLOv5 ONNX 转换示例：<https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/examples/rknn_yolov5_demo/convert_rknn_demo/yolov5/onnx2rknn.py>

```python
# ===== RKNN INT8 量化公式演示 =====
# 这里只演示量化数学关系，不依赖 rknn-toolkit2，也不会执行模型转换。

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy 未安装，跳过数组演示")


def calculate_asymmetric_quant_params(min_val, max_val, qmin=-128, qmax=127):
    """根据浮点范围计算非对称 INT8 量化参数。"""
    if max_val <= min_val:
        raise ValueError("max_val 必须大于 min_val")

    scale = (max_val - min_val) / float(qmax - qmin)
    zero_point = round(qmin - min_val / scale)
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point


if NUMPY_AVAILABLE:
    fp32_values = np.array([-1.2, -0.3, 0.0, 0.7, 2.4], dtype=np.float32)
    scale, zero_point = calculate_asymmetric_quant_params(
        float(fp32_values.min()),
        float(fp32_values.max()),
    )

    int8_values = np.round(fp32_values / scale + zero_point)
    int8_values = np.clip(int8_values, -128, 127).astype(np.int8)
    restored_values = scale * (int8_values.astype(np.float32) - zero_point)

    print("FP32 原始值:", fp32_values)
    print("scale:", scale)
    print("zero_point:", zero_point)
    print("INT8 量化值:", int8_values)
    print("反量化近似值:", restored_values)
    print("量化误差:", fp32_values - restored_values)
else:
    print("公式：q = round(x / scale + zero_point)")
    print("公式：x ≈ scale * (q - zero_point)")
```
