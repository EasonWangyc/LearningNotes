# Triton 学习笔记：一手来源摘录

> 范围：为 `notes.md` 和后续小型 demo 准备事实依据；只采用 Triton 官方文档与 `triton-lang/triton` 官方仓库。本文不表示当前环境已经安装或验证了 Triton。
>
> 核对日期：2026-08-02。Triton 的 `main` 文档随版本演进，实际编写/运行 demo 时应再次对照所安装版本的 API。

## 1. 定位与最小编程模型

- Triton 是面向并行编程的语言与编译器，目标是在现代 GPU 上以 Python 为入口高效编写自定义 DNN 计算 Kernel。[官方首页](https://triton-lang.org/main/index.html)
- 与 CUDA 以标量线程表达、线程块承载并行不同，Triton 的核心抽象是**一个 program instance 处理一个数据块（tile）**。程序之间由 launch `grid` 划分工作；在 kernel 内用 `tl.program_id(axis)` 取得当前实例编号，并用 `tl.arange` 构造块内偏移。[编程模型导论](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)；[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- 最小 Kernel 通常由 `@triton.jit`、指针参数、`tl.load` / `tl.store`、块大小元参数和 host 端 `kernel[grid](...)` 组成。`@triton.jit` 的函数由 Triton 编译器 JIT 编译并在 GPU 运行；其可访问范围限于 Python 基本类型、`triton` 包内置对象、函数参数和其他 JIT 函数。[`triton.jit` API](https://triton-lang.org/main/python-api/generated/triton.jit.html)
- `BLOCK_SIZE: tl.constexpr` 表示编译期元参数；官方向量加法教程将其用于 shape 值。它适合块大小、算法分支等影响生成代码的配置，而非每次调用都频繁变化的普通运行时数据。[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

## 2. 数据块、指针与边界

- 一维入门模式：`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`，用 `offsets < n_elements` 生成尾块 mask，再将该 mask 同时传给 `tl.load` 和 `tl.store`。这样最后一个不满块不会越界。[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- `tl.load` / `tl.store` 支持标量指针、N 维指针张量和 block pointer；普通指针张量的掩码会广播到其 shape。[`tl.load` API](https://triton-lang.org/main/python-api/generated/triton.language.load.html)；[`tl.store` API](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
- `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` 从父张量元数据构造一个块指针；其中 `shape`、`strides` 描述父张量，`offsets` 是块起点，`block_shape` 是本次访问块形状，`order` 描述原始数据格式的顺序。[`tl.make_block_ptr` API](https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html)
- 使用 block pointer 时，`tl.load` 的 `mask` 与 `other` 必须为 `None`，`tl.store` 的 `mask` 也必须为 `None`；越界处理改由 `boundary_check`（以及 load 的 `padding_option`）表达。循环处理相邻 tile 可用 `tl.advance(block_ptr, offsets)` 移动块指针。[`tl.load` API](https://triton-lang.org/main/python-api/generated/triton.language.load.html)；[`tl.store` API](https://triton-lang.org/main/python-api/generated/triton.language.store.html)；[`tl.advance` API](https://triton-lang.org/main/python-api/generated/triton.language.advance.html)
- 但 `make_block_ptr` 已不应作为新 demo 的主路径：官方 v3.7.0 release notes 说明其调用会产生 deprecation warning。因此保留本节用于阅读存量示例和理解 `boundary_check`；新代码应先沿用当前版本教程/API 推荐的访问方式，并在定稿前复核版本迁移说明。[官方 Releases（v3.7.0）](https://github.com/triton-lang/triton/releases/tag/v3.7.0)

## 3. Layout：当前学习边界

- 标准 Triton 入门 kernel 可以先从 `tl.arange`、指针张量和 block pointer 学起；源码/教程中的 `num_warps`、块大小及访问方式是先验证后调优的关键配置。[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；[矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- 显式 tensor layout 是实验性 **Gluon** 的更低层模型：每个 tensor 必须带 layout，layout 定义 tile 元素如何分配到 CTA、warp、lane 和寄存器；`BlockedLayout` 是常见类型。它会影响全局内存访问、跨线程通信和 shared-memory bank conflict，进而显著影响性能。[Gluon Tensor Layouts 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)
- 因此，第一阶段不要把 **block pointer** 与 **Gluon layout** 混为同一概念：前者描述从父张量访问一个逻辑块的指针；后者描述 tile 元素到线程/寄存器层级的映射。这是根据上述两个官方 API/教程作出的术语划分，后续学习 layout 时应明确标注为 Gluon 主题。

## 4. 安装、平台与首次验证

- 官方稳定版安装命令为 `pip install triton`；当前安装页写明二进制 wheel 覆盖 CPython 3.10–3.14。源码安装需要仓库的 `python/requirements.txt`，若系统未安装 LLVM，构建脚本会下载官方 LLVM 静态库。[安装文档](https://triton-lang.org/main/getting-started/installation.html)
- 官方仓库当前 Compatibility 小节列出 Linux，NVIDIA GPU Compute Capability 8.0+，AMD GPU ROCm 6.2+；CPU 后端仍标为 under development。因此 Windows 主机不应把 `pip install triton` 的成功安装预设为官方支持的可运行组合，应先确认实际 wheel、Python、PyTorch/驱动和 GPU 组合。[官方仓库 README](https://github.com/triton-lang/triton#compatibility)
- 源码仓库的全量测试需 GPU：`make dev-install` 后运行 `make test`；无 GPU 可运行 `make test-nogpu`。这些是源码树维护/验证命令，不等同于一个已安装 wheel 的用户 demo。[安装文档](https://triton-lang.org/main/getting-started/installation.html)

## 5. 正确性、调试与性能测量

- 正确性基线应由相同输入上的 PyTorch（或其他可信实现）给出；官方向量加法教程直接比较结果并报告最大绝对误差。`triton.testing.assert_close` 是 testing 模块提供的近似相等断言。[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；[`triton.testing` API](https://triton-lang.org/main/python-api/triton.testing.html)
- 编译期检查可使用 `tl.static_print`、`tl.static_assert`；设备运行期可使用 `tl.device_print`、`tl.device_assert`。其中 `device_assert` 仅在 `TRITON_DEBUG=1` 时执行。[调试文档](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- 设置 `TRITON_INTERPRET=1` 会绕过编译，以 NumPy 等价操作在 CPU 上逐个、顺序模拟 program instance，适合观察中间值和配合 `pdb` 定位问题。解释器当前不支持 `bfloat16` 运算和间接访存（先 load 指针、再以该指针 load）的模式，因此解释器通过不代表 GPU 编译路径必然可用。[调试文档](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- NVIDIA GPU 上可用 `compute-sanitizer` 检查数据竞争与内存访问问题；这是 GPU 真路径调试的一部分。[调试文档](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- `triton.testing.do_bench(fn, warmup=25, rep=100, ...)` 以毫秒返回运行时间；可请求分位数。`triton.testing.perf_report` / `Benchmark` 用于组织不同问题规模或 provider 的曲线/表格。官方示例先 warm-up、再对同一输入规模比较 PyTorch 与 Triton，并按总读写字节和耗时换算 GB/s。[`do_bench` API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)；[向量加法基准教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- `@triton.autotune` 会为同一个 `key` 对应的输入形状测试多个 `triton.Config`；被调 kernel 会被重复执行。若 kernel 原地更新输出，应使用 `reset_to_zero` 或 `restore_value` 避免调优过程改变结果。[`triton.autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)

## 6. 用于第一版 demo 的可验证闭环

建议最先实现 **Vector Add 或 ReLU**，而不是先做矩阵乘：两者都足以覆盖 program/grid、`tl.arange`、mask、load/store、尾块和基准对照，同时把数学正确性与性能计量分开。

```text
固定随机种子和若干不规则长度
        ↓
PyTorch reference 与 Triton 输出做 assert_close / 最大误差检查
        ↓
TRITON_INTERPRET=1 检查索引和边界（受解释器限制约束）
        ↓
GPU 路径运行；必要时 compute-sanitizer
        ↓
分别 warm-up 后，以相同 shape、dtype、设备和同步口径调用 do_bench
        ↓
记录中位数/分位数、GB/s、Triton 版本和 GPU 信息
```

上图中的 demo 路径是对官方向量加法、调试和 testing 文档的学习顺序建议，不是官方规定的唯一流程。
