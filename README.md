# 📚 Learning Notes

> 面向长期积累的技术学习笔记，覆盖编程语言、系统工具和 AI（机器学习、大语言模型与推理部署）。

---

## 📂 目录结构

### 💻 编程语言

| 主题 | 内容 |
|------|------|
| 🔵 [C](./Languages/C/notes.md) | 数据类型、指针、内存、数组、结构体和标准库 |
| 🔷 [C++](./Languages/Cpp/notes.md) | 类型系统、数据结构、面向对象和现代 C++ |
| 🐍 [Python](./Languages/Python/notes.md) | Python 基础与常用数据处理、深度学习和 Web 子模块 |

### 🛠️ 系统与工具

| 主题 | 内容 |
|------|------|
| 🐧 [Linux](./Tools/Linux/notes.md) | 文件、进程、线程、IPC、网络和 Vim/vi/nano |
| 🐚 [Shell](./Tools/Shell/notes.md) | Shell 脚本、命令行和自动化 |
| 🏗️ [CMake](./Tools/CMake/notes.md) | C/C++ 构建系统与 GCC 编译流程 |
| 🧪 [Conda](./Tools/Conda/notes.md) | Python 环境与包管理 |
| ⚡ [uv](./Tools/UV/notes.md) | Python 项目、虚拟环境与依赖管理 |
| 🐳 [Docker](./Tools/Docker/notes.md) | 容器、镜像、Compose、GPU 和大模型服务 |
| 🌿 [Git](./Tools/Git/notes.md) | 版本控制、分支、提交和协作工作流 |
| 🔐 [SSH](./Tools/SSH/notes.md) | 远程连接、密钥认证、端口转发和内网穿透 |

### 🤖 AI

#### 机器学习基础

| 主题 | 内容 |
|------|------|
| 📊 [ML](./AI/ML/notes.md) | 经典机器学习、数据处理、模型训练和评估 |
| 🧠 [DL](./AI/DL/notes.md) | 深度学习、神经网络、损失函数和训练流程 |
| 🎮 [RL](./AI/RL/notes.md) | 强化学习、价值函数、策略优化和环境交互 |

#### 深度学习框架

| 主题 | 内容 |
|------|------|
| 🐍 [PyTorch](./AI/Framework/Pytorch/notes.md) | 张量操作、自动求导、模型训练与部署 |

#### 大语言模型

| 主题 | 内容 |
|------|------|
| 🧩 [LLM 总索引](./AI/LLM/notes.md) | 按基础、训练、推理和应用组织的大语言模型笔记 |
| 🧱 [LLM 基础](./AI/LLM/Basics/notes.md) | Tokenizer、Embedding、Transformer、Attention 和 MoE |
| 🏋️ [LLM 训练](./AI/LLM/Training/notes.md) | LoRA、数据集、权重装载和并行训练 |
| 🚀 [LLM 推理](./AI/LLM/Inference/notes.md) | 解码、Prefill、KV Cache、PagedAttention 和推理优化 |
| 🧰 [LLM 应用](./AI/LLM/Applications/notes.md) | RAG、检索、重排和 Harness Engineering |
| 🤖 [Agent](./AI/LLM/Applications/Agent/notes.md) | Agent Loop、Context、Tools、Skills、Hooks、MCP 和 Subagents |

#### AI 基础设施与部署

| 主题 | 内容 |
|------|------|
| 🧱 [AI Infra 总索引](./AI/Infra/notes.md) | 硬件、内存、通信、运行时、并行训练、量化和服务基础 |
| 🔥 [CUDA](./AI/Infra/CUDA/notes.md) | GPU 编程、线程层级、内存和性能优化 |
| 📦 [ONNX](./AI/Infra/ONNX/notes.md) | 模型交换格式、导出和 ONNX Runtime |
| 🚀 [TensorRT](./AI/Infra/TensorRT/notes.md) | Engine 构建、精度、动态形状和推理部署 |
| 🔷 [Triton](./AI/Infra/Triton/notes.md) | GPU Kernel、算子融合、性能测量和调优 |
| ⚡ [vLLM](./AI/Infra/vLLM/notes.md) | 在线推理服务、PagedAttention、Continuous Batching 和量化 |

> 代码实验（GQA、PagedAttention、tiled GEMM、mini-LLM）位于 [`AI/from_scratch/`](./AI/from_scratch/)。

---

## 🛠️ 环境

- 主笔记格式：Markdown `.md`
- 实验与调试格式：[Jupyter Notebook](https://jupyter.org/) `.ipynb`
- 代码块按主题放入对应笔记，避免创建零散脚本文件
- 图片与视频资源放在对应主题的 `resources/` 目录

## 📝 说明

笔记内容持续整理，主题之间通过 README、专题索引和相对链接连接。
