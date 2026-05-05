# AGENTS.md — LearningNotes 开发指南

本文件用于指导 AI Agent（如 OpenAI Codex）在该仓库中进行开发和内容维护。

---

## 仓库概述

这是一个**个人技术速成笔记仓库**，以 Markdown（`.md`）作为长期维护与归档主格式。Jupyter Notebook（`.ipynb`）只保留给需要 Python kernel 调试、交互实验、绘图或模型推理验证的主题。内容覆盖编程语言、系统工具、AI/ML、大模型、推理部署、机器人等领域。目标是"速查速学"，而非面面俱到。

---

## 目录结构

```
LearningNotes/
├── readme.md                  ← 主索引，维护所有笔记的导航链接
├── C/notes.md                 ← 主笔记
├── Cpp/notes.md
├── CMake/notes.md
├── Conda/notes.md
├── Docker/notes.md
├── Git/notes.md
├── Linux/notes.md
├── Shell/notes.md
├── Cuda/
│   ├── notes.md
│   └── notes.ipynb            ← 需要 Python 调试/验证时保留
├── ONNX/
│   ├── notes.md
│   └── notes.ipynb
├── Python/                    ← 子主题通常保留 md + notebook
│   ├── notes.md
│   ├── notes.ipynb
│   ├── Pytorch.md
│   └── Pytorch.ipynb
├── vLLM/
│   ├── notes.md
│   └── notes.ipynb
└── ...
```

---

## Markdown 主笔记规范

### 适用范围

- **优先维护 `.md`**：概念解释、面试速记、命令速查、对比表格、流程图、坑点总结、部署步骤。
- **谨慎维护 `.ipynb`**：只用于交互式调试、tensor shape 验证、绘图、模型推理实验、需要 kernel 执行的学习过程。
- **不要假设每个主题都有 notebook**：C、C++、CMake、Conda、Docker、Git、Linux、Shell 等偏概念/命令型主题通常只保留 `notes.md`。
- AI Agent 修改笔记时，默认修改 `.md`；只有用户明确要求可运行 notebook、需要调试代码，或当前主题已经保留 notebook 时，才修改 `.ipynb`。

### 推荐结构

每个主题的 `notes.md` 通常包含：

```markdown
# 主题名称

## 内容概览
- 核心概念 A
- 核心概念 B
- 实践工具 C

## 1. 理论详解

## 2. 常用命令 / 代码模板

## 3. 面试速记 / 常见问题

## 4. 常见坑点
```

格式要求：

- **语言**：中文说明为主，代码标识符使用英文。
- **表格**：对比类内容优先使用 Markdown 表格。
- **代码块语言标注**：必须写明 `bash`、`python`、`yaml`、`dockerfile`、`text` 等。
- **结构图**：优先使用 ASCII 图或 Mermaid；简单流程可用 `text` 代码块。
- **图片**：引用同目录 `resources/` 子目录，格式：
  ```markdown
  <p align="center">
    <img src="resources/1.png" width="60%">
  </p>
  ```
- **内容风格**：面向速查，避免写成长篇教材；优先保留关键结论、必要背景、最小可用示例。

---

## Notebook 内容规范

### 适用范围

Notebook 用于实验和调试，不再作为长期归档主格式。保留 notebook 的主题通常包括：

- Python 基础与子模块：NumPy、Pandas、Matplotlib、PyTorch、TensorFlow、Scikit-learn、FastAPI 等
- AI/ML/RL/DL 相关实验
- ONNX、CUDA、TensorRT、vLLM、VLM、VLA 等需要 Python 验证或模型推理的主题
- ROS2 等需要代码或命令交互验证的主题

### 推荐 Cell 结构

Notebook 不要求承载完整理论笔记，理论归档应写入 `.md`。Notebook 建议保留以下内容：

1. **Cell 1 — 标题与概览**（Markdown）

   ```markdown
   # 主题名称

   ## 内容概览
   - 核心概念 A
   - 核心概念 B
   - 实践工具 C
   ```

2. **Cell 2 — 调试目标 / 实验说明**（Markdown）

   简要说明要验证什么，例如 tensor shape、API 调用、模型导出、推理结果、性能对比等。

3. **Cell 3 — 可运行代码**（Python）

   可运行的 Python 代码，要求：
   - 聚焦调试或验证目标，不把 notebook 写成大段教材
   - 对依赖库不可用的情况做 `try/except ImportError` 降级处理
   - 添加清晰的 `# ===== 章节名 =====` 分隔注释
   - Windows 环境兼容（避免硬编码 Linux 路径，bash 命令用 `subprocess` 封装并判断平台）

### Notebook 格式约定

- **语言**：全中文注释和说明，代码本身用英文标识符
- **表格**：对比类内容优先用 Markdown 表格呈现
- **代码块语言标注**：必须写明（`bash`、`python`、`yaml`、`dockerfile` 等）
- **图片**：引用 `resources/` 子目录下的图片，格式：
  ```markdown
  <p align="center">
    <img src="resources/1.png" width="60%">
  </p>
  ```
- **Cell 数量**：保持精简，优先保留调试和验证所需内容；归档型说明必须沉淀到 `.md`

---

## readme.md 维护规范

- 新增主题时，必须同步更新 `readme.md` 中对应分类的表格
- 表格格式：`| emoji [主题](./目录/notes.md) | 内容简述 | Markdown / Jupyter Notebook |`
- 如果主题有配套 notebook，可在对应目录保留同名 `.ipynb`，但 `readme.md` 主链接优先指向 `.md`
- 不要因为新增主题而默认创建 notebook；只有明确需要 Python 调试或交互实验时才创建
- 分类已固定（见下），不要随意新增顶级分类：
  - 💻 编程语言
  - 🛠️ 系统与工具
  - 🤖 AI / 机器学习基础（ML / DL / RL）
  - 🌐 大模型 / 多模态（LLM → VLM → VLA）
  - ⚡ 推理与部署（CUDA / ONNX / TensorRT / vLLM）
  - 🤖 机器人（ROS2）

---

## 新增主题流程

1. 在根目录创建新文件夹（如 `Triton/`）
2. 在文件夹内创建 `notes.md`，结构遵循"Markdown 主笔记规范"
3. 如需 Python 调试、绘图、模型推理或交互实验，再创建 `notes.ipynb`
4. 如有图片资源，放入 `Triton/resources/`
5. 更新 `readme.md` 对应分类表格，补充条目

---

## 修改已有笔记

- **追加内容**：优先追加到对应主题的 `.md` 章节中；若是实验过程，再追加到 notebook
- **修订内容**：直接编辑 `.md` 对应章节，保持原有知识结构不变
- **Notebook 修订**：只保留可运行调试逻辑；如果发现 notebook 中有大量理论说明，优先整理到 `.md`
- **删除 notebook**：用户可能会删除不再需要调试的 notebook；不要恢复已删除的 notebook，除非用户明确要求
- **禁止**：删除已有 notebook 的有效实验内容、随意更改 `readme.md` 的分类体系、将多个主题混进同一文件

---

## 代码风格要求

```python
# ✅ 推荐写法
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ℹ️  PyTorch 未安装，跳过相关示例")

# ===== 章节标题 =====
print("=" * 50)
print("1. 核心概念演示")
print("=" * 50)

if TORCH_AVAILABLE:
    # 实际代码
    ...
else:
    # Python 等效逻辑 / 占位提示
    print("需要安装 torch 才能运行此示例")
```

---

## 禁止行为

- 不要创建零散 `.py` 作为笔记；需要代码演示时放入 `.md` 代码块或 notebook
- 不要在 notebook 中插入大段归档型说明或无注释代码块
- 不要修改 `readme.md` 的整体结构（知识结构说明、分类体系）
- 不要新建或恢复 `others.md` / `others.ipynb` 来写主题性笔记；零散知识点应尽量归并到对应主题
- 不要在代码 Cell 中执行网络请求或下载模型权重（会导致 CI 超时）
