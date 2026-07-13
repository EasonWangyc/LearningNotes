# 📚 Learning Notes

> 个人技术学习笔记仓库，涵盖编程语言、工具链、AI/ML 等多个领域。

---

## 📂 目录结构

### 💻 编程语言

| 分类 | 内容 |
|------|------|
| 🔵 [C](./C/notes.md) | C 语言基础：指针、内存、数据结构 |
| 🔷 [C++](./Cpp/notes.md) | C++ 语言进阶 |
| 🐍 [Python](./Python/notes.md) | 语法、NumPy、Matplotlib、Pandas、PyTorch、TensorFlow、FastAPI、Scikit-learn |

### 🛠️ 系统与工具

| 分类 | 内容 |
|------|------|
| 🐧 [Linux](./Linux/notes.md) | Linux 系统使用与命令 |
| 🐚 [Shell](./Shell/notes.md) | Shell 脚本与命令行 |
| 🏗️ [CMake](./CMake/notes.md) | 跨平台构建系统 |
| 🧪 [Conda](./Conda/notes.md) | Python 环境与包管理 |
| ⚡ [UV](./UV/notes.md) | Python 项目、虚拟环境与依赖管理 |
| 🐳 [Docker](./Docker/notes.md) | 容器化与镜像管理 |
| 🌿 [Git](./Git/notes.md) | 版本控制与协作工作流 |
| 🧱 [Infra](./Infra/notes.md) | AI 基础设施：硬件、分布式训练、推理服务与性能观测 |
| 🔐 [SSH](./SSH/notes.md) | 远程连接、密钥认证、端口转发与内网穿透 |

### 🤖 AI / 机器学习基础

> ML、DL、RL 是 AI 的三大学习范式，各有侧重。

| 分类 | 内容 |
|------|------|
| 📊 [ML](./ML/notes.md) | 经典机器学习（SVM、决策树、聚类、降维等） |
| 🧠 [DL](./DL/notes.md) | 深度学习基础（CNN、RNN、Transformer 等） |
| 🎮 [RL](./RL/notes.md) | 强化学习（Q-Learning、PPO、SAC 等） |

### 🌐 大模型 / 多模态

> LLM → VLM → VLA，能力逐步扩展：纯语言 → 视觉语言 → 视觉语言动作。

| 分类 | 内容 |
|------|------|
| 💬 [LLM](./LLM/notes.md) | 大语言模型（GPT、LLaMA、Qwen 等） |
| 🧩 [Agent](./Agent/notes.md) | AI Agent 工具体系（Context、Tools、Skills、Hooks、MCP、Subagents 等） |
| 👁️ [VLM](./VLM/notes.md) | 视觉语言模型（CLIP、LLaVA、InternVL、Qwen-VL 等） |
| 🦾 [VLA](./VLA/notes.md) | 视觉语言动作模型（RT-2、OpenVLA、π0 等） |

### ⚡ 推理与部署

| 分类 | 内容 |
|------|------|
| 🔥 [CUDA](./Cuda/notes.md) | NVIDIA CUDA 并行计算 |
| 📦 [ONNX](./ONNX/notes.md) | 模型交换格式与跨框架推理 |
| 🚀 [TensorRT](./TensorRT/notes.md) | NVIDIA 高性能推理引擎 |
| ⚡ [vLLM](./vLLM/notes.md) | 大语言模型高效推理框架（PagedAttention） |

### 🤖 机器人

| 分类 | 内容 |
|------|------|
| 🤖 [ROS2](./ROS2/notes.md) | 机器人操作系统 ROS 2 |

---

## 🗂️ Python 子模块

| 文件 | 内容 |
|------|------|
| [notes.md](./Python/notes.md) | Python 基础语法 |
| [Numpy&Matplotlib&Pandas.md](./Python/Numpy%26Matplotlib%26Pandas.md) | 数据处理与可视化 |
| [Pytorch.md](./Python/Pytorch.md) | 深度学习框架 PyTorch |
| [TensorFlow.md](./Python/TensorFlow.md) | 深度学习框架 TensorFlow |
| [Scikit_learn.md](./Python/Scikit_learn.md) | 机器学习库 Scikit-learn |
| [FastAPI.md](./Python/FastAPI.md) | Web 框架 FastAPI |

---

## 🧩 知识结构说明

### AI 三大学习范式

```
ML（机器学习）—— 从数据中学习规律，经典算法为主
  ├── DL（深度学习）—— 以神经网络为核心，ML 的子集
  └── RL（强化学习）—— 通过与环境交互学习策略，可与 DL 结合（Deep RL）
```

### 大模型演进路线

```
LLM（大语言模型）
  └── 纯文本理解与生成

VLM（视觉语言模型）= LLM + Vision Encoder
  └── 图文理解、视觉问答、图像描述

VLA（视觉语言动作模型）= VLM + Action Head
  └── 机器人端到端感知-规划-执行
```

### 推理部署工具链

```
训练完成的模型
  ├── ONNX 导出（框架无关的中间格式）
  │     └── TensorRT 优化（NVIDIA GPU 高性能推理）
  └── vLLM（专为 LLM/VLM 设计的高吞吐推理服务）
        └── 依赖 CUDA 并行计算底层
```

---

## 🛠️ 环境

- 主笔记格式：Markdown `.md`
- 实验与调试格式：[Jupyter Notebook](https://jupyter.org/) `.ipynb`
- 推荐使用 [VS Code](https://code.visualstudio.com/) 进行 Markdown 维护，并配合 Jupyter 扩展或 JupyterLab 调试 notebook

---

## 📝 说明

本仓库为个人学习记录，持续更新中。笔记内容来源包括官方文档、菜鸟教程、各类课程等，仅供学习参考。
