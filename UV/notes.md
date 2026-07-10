# uv

uv 是 Astral 使用 Rust 编写的 Python 包管理和项目管理工具。它可以理解为一个把 `pip`、`virtualenv`、`pip-tools`、`pipx`、部分 `pyenv` 和 `poetry` 工作流合在一起的工具：既能创建虚拟环境，也能安装依赖、锁定依赖、运行命令、管理 Python 版本和运行一次性命令行工具。

对于学习机器学习来说，uv 最直接的价值是：不依赖 Conda，也可以在项目目录里快速创建 `.venv`，安装 `numpy`、`pandas`、`scikit-learn`、`matplotlib`、`jupyter`、`torch` 等依赖，然后用 VS Code 或 Jupyter Notebook 选择这个环境作为 kernel。

## uv 解决的问题

Python 环境管理里经常混在一起的其实是几件事：

| 问题 | 说明 | 常见工具 |
| --- | --- | --- |
| Python 版本 | 使用 Python 3.10、3.11 还是 3.12 | pyenv、conda、uv |
| 虚拟环境 | 隔离不同项目的包 | venv、virtualenv、conda、uv |
| 包安装 | 安装第三方库 | pip、conda、uv |
| 依赖声明 | 记录项目需要哪些包 | requirements.txt、pyproject.toml |
| 依赖锁定 | 固定完整依赖树版本 | pip-tools、poetry.lock、uv.lock |
| 命令运行 | 在项目环境中运行脚本或工具 | python、pipx、poetry run、uv run |
| 一次性工具 | 临时运行 ruff、black、jupyter 等命令 | pipx、uvx |

uv 的思路是把这些能力尽量统一到一个命令体系里。

```text
Python 项目
  ├── pyproject.toml     # 项目元数据和直接依赖
  ├── uv.lock            # uv 解析出的完整依赖锁
  ├── .python-version    # 当前项目默认 Python 版本
  └── .venv/             # 当前项目虚拟环境
```

## 安装 uv

### Windows PowerShell

官方安装脚本：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后检查版本：

```powershell
uv --version
```

如果当前环境不允许执行网络脚本，可以改用 `pipx`、`pip`、`winget` 或直接下载二进制文件。具体方式取决于操作系统、终端权限和网络策略。

```powershell
pip install uv
```

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

### 更新 uv

如果是官方独立安装器安装的 uv，可以使用：

```bash
uv self update
```

如果是 `pip` 安装的 uv，则使用：

```bash
pip install -U uv
```

## uv 的两种主要使用方式

uv 有两条常见路线：

| 模式 | 核心命令 | 适合场景 |
| --- | --- | --- |
| 项目模式 | `uv init`、`uv add`、`uv run`、`uv lock`、`uv sync` | 新项目、长期维护项目、希望有 `pyproject.toml` 和 `uv.lock` |
| pip 兼容模式 | `uv venv`、`uv pip install`、`uv pip sync` | 老项目、只有 `requirements.txt`、临时虚拟环境 |

学习和维护 ML 项目环境时，更推荐项目模式；临时运行脚本或复现已有仓库时，可以先用 pip 兼容模式。

## 项目模式

项目模式是 uv 最推荐的使用方式。它把项目依赖写入 `pyproject.toml`，把完整解析结果写入 `uv.lock`，然后用 `.venv` 保存当前项目环境。

### 创建项目

创建新项目：

```bash
uv init ml-study
cd ml-study
```

在当前目录初始化：

```bash
mkdir ml-study
cd ml-study
uv init
```

典型结构：

```text
ml-study/
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

第一次运行 `uv run`、`uv sync` 或 `uv lock` 后，uv 会创建 `.venv` 和 `uv.lock`：

```text
ml-study/
├── .venv/
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

### pyproject.toml

`pyproject.toml` 是 Python 现代项目的配置入口。uv 会把项目名、Python 版本约束和依赖写在这里。

```toml
[project]
name = "ml-study"
version = "0.1.0"
description = "Personal machine learning study environment"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []
```

添加依赖后会变成：

```toml
[project]
name = "ml-study"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0.0",
    "pandas>=2.2.0",
    "scikit-learn>=1.5.0",
]
```

`pyproject.toml` 记录的是“直接依赖”，也就是项目明确声明需要的库。

### uv.lock

`uv.lock` 记录完整依赖树的精确版本。比如项目直接安装的是 `scikit-learn`，但它还会依赖 `numpy`、`scipy`、`joblib`、`threadpoolctl` 等包，uv 会把解析后的完整结果写进锁文件。

```text
pyproject.toml   记录直接依赖：scikit-learn
uv.lock          记录完整依赖树：scikit-learn + numpy + scipy + ...
.venv/           实际安装出来的环境
```

通常做法：

```text
提交到 Git：
- pyproject.toml
- uv.lock

不提交到 Git：
- .venv/
```

这样迁移到新机器后，只要执行 `uv sync`，就能尽量复现同一个环境。

### 添加依赖

添加普通依赖：

```bash
uv add numpy pandas matplotlib scikit-learn
```

添加指定版本：

```bash
uv add "scikit-learn>=1.5"
uv add "numpy==2.0.2"
```

从 `requirements.txt` 导入：

```bash
uv add -r requirements.txt
```

从 Git 安装：

```bash
uv add "example-pkg @ git+https://github.com/user/example-pkg"
```

### 添加开发依赖

开发依赖通常只服务于调试、格式化、测试或 notebook，不一定属于项目运行时依赖。

```bash
uv add --dev ipykernel pytest ruff
```

常见写法：

```toml
[dependency-groups]
dev = [
    "ipykernel>=6.29.0",
    "pytest>=8.0.0",
    "ruff>=0.8.0",
]
```

文档型学习项目里，`ipykernel`、`jupyterlab`、`pytest` 更适合放在开发依赖；`numpy`、`pandas`、`scikit-learn` 这类示例代码直接 import 的库更适合放在普通依赖。

### 移除依赖

```bash
uv remove pandas
uv remove --dev pytest
```

移除依赖时，uv 会同步修改 `pyproject.toml` 和 `uv.lock`。

### 运行命令

`uv run` 会在项目环境里执行命令。即使没有手动激活 `.venv`，也会使用当前项目环境。

```bash
uv run python main.py
uv run python -c "import sklearn; print(sklearn.__version__)"
uv run pytest
uv run ruff check .
```

如果当前环境还没有同步，`uv run` 会先自动 lock 和 sync，然后再执行命令。

### 同步环境

根据 `uv.lock` 安装依赖：

```bash
uv sync
```

只检查锁文件是否最新，不自动更新：

```bash
uv lock --check
uv run --locked python main.py
```

只使用现有锁文件，不检查是否过期：

```bash
uv run --frozen python main.py
```

不自动同步环境：

```bash
uv run --no-sync python main.py
```

`uv sync` 默认更像“精确同步”：环境中不在锁文件里的包可能会被移除。临时保留额外包可以使用：

```bash
uv sync --inexact
```

### 升级依赖

更新锁文件中的包版本：

```bash
uv lock --upgrade
```

只升级某一个包：

```bash
uv lock --upgrade-package scikit-learn
```

更新后同步环境：

```bash
uv sync
```

## pip 兼容模式

如果不想创建完整 uv 项目，只想像 `venv + pip` 一样使用，可以用 uv 的 pip 兼容模式。

### 创建虚拟环境

在当前目录创建 `.venv`：

```bash
uv venv
```

指定 Python 版本：

```bash
uv venv --python 3.11
```

指定虚拟环境目录：

```bash
uv venv .venv-ml
```

### 激活虚拟环境

Windows PowerShell：

```powershell
.venv\Scripts\activate
```

Linux / macOS：

```bash
source .venv/bin/activate
```

退出虚拟环境：

```bash
deactivate
```

不激活也可以直接使用 `uv run` 或 `uv pip`，但激活后在 VS Code、Jupyter 和终端里更直观。

### 安装包

```bash
uv pip install numpy pandas matplotlib scikit-learn
uv pip install "flask[dotenv]"
uv pip install "ruff>=0.8.0"
```

安装 PyTorch CPU 版本时，`--index-url` 是安装命令的参数，不要包进依赖字符串里：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 查看环境

```bash
uv pip list
uv pip show numpy
uv pip freeze
```

### 从 requirements.txt 安装

```bash
uv pip install -r requirements.txt
```

如果想让环境严格等于 `requirements.txt`：

```bash
uv pip sync requirements.txt
```

`install -r` 是“在已有环境上安装这些包”，`sync` 是“把环境同步成这个文件描述的状态”。

### 生成 requirements.txt

从输入文件生成锁定版本：

```bash
uv pip compile requirements.in -o requirements.txt
```

示例：

```text
requirements.in
├── numpy
├── pandas
└── scikit-learn

requirements.txt
├── numpy==...
├── pandas==...
├── scikit-learn==...
├── scipy==...
├── joblib==...
└── threadpoolctl==...
```

## Python 版本管理

uv 可以安装和选择 Python 版本。

安装多个版本：

```bash
uv python install 3.10 3.11 3.12
```

查看可用 Python：

```bash
uv python list
```

为当前项目固定 Python 版本：

```bash
uv python pin 3.11
```

这会生成或更新 `.python-version`：

```text
3.11
```

之后创建 `.venv` 时，uv 会优先使用这个版本。

### uv 与 pyenv 的区别

| 维度 | pyenv | uv |
| --- | --- | --- |
| 核心作用 | 管理多个 Python 解释器 | 管理 Python 版本、项目、依赖和虚拟环境 |
| 是否管理依赖 | 不管理 | 管理 |
| 是否生成 lockfile | 不生成 | 生成 `uv.lock` |
| 使用方式 | `pyenv install`、`pyenv local` | `uv python install`、`uv python pin`、`uv sync` |

pyenv 更像“Python 解释器版本管理器”；uv 更像“Python 项目环境管理器”。

## Jupyter / ipynb 使用方式

Jupyter / ipynb 可以直接使用 uv 创建的 `.venv`。这种方式不依赖 Conda，适合在受限环境中维护可复现的 Python 项目环境。

### 项目内启动 Jupyter

在项目中添加常用 ML 依赖：

```bash
uv init ml-notes
cd ml-notes
uv add numpy pandas matplotlib scikit-learn
uv add --dev ipykernel
```

直接启动 Jupyter Lab：

```bash
uv run --with jupyter jupyter lab
```

`--with jupyter` 的意思是：临时给这次命令提供 Jupyter 本身，但 notebook 里 import 的项目依赖仍然来自项目环境。

### 给项目创建 kernel

如果希望 VS Code 或 Jupyter 稳定识别当前项目环境，建议安装 `ipykernel`：

```bash
uv add --dev ipykernel
```

Linux / macOS：

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=ml-notes
```

Windows PowerShell：

```powershell
uv run ipython kernel install --user --env VIRTUAL_ENV "$PWD\.venv" --name=ml-notes
```

之后在 notebook 中选择 `ml-notes` kernel。

### VS Code 中选择 uv 环境

在 VS Code 中打开项目目录后：

```text
Command Palette
  ↓
Python: Select Interpreter
  ↓
选择 .venv\Scripts\python.exe 或 .venv/bin/python
```

创建 notebook 时，如果提示选择 kernel，就选择这个 `.venv` 对应的 Python。

Windows 路径通常是：

```text
.venv\Scripts\python.exe
```

Linux / macOS 路径通常是：

```text
.venv/bin/python
```

### Notebook 内安装包

更推荐在终端中使用：

```bash
uv add seaborn
```

这样依赖会写入 `pyproject.toml` 和 `uv.lock`，迁移到新机器后也更容易复现。

如果只是临时试一下：

```python
!uv pip install seaborn
```

这种方式可能不会更新项目依赖声明。长期维护的文档型项目建议尽量使用 `uv add`。

## ML 学习项目示例

假设要给机器学习项目创建一个可运行环境：

```bash
uv init ml-study
cd ml-study
uv python pin 3.11
uv add numpy pandas matplotlib seaborn scikit-learn jupyterlab
uv add --dev ipykernel
uv run python -c "import sklearn; print(sklearn.__version__)"
```

创建 notebook：

```bash
uv run --with jupyter jupyter lab
```

或者只在 VS Code 中使用 kernel：

```bash
uv add --dev ipykernel
```

一个简单验证：

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("accuracy:", model.score(X_test, y_test))
```

如果这段代码能在 notebook 中运行，说明当前 uv 环境已经适合学习 ML 基础内容。

## PyTorch 与深度学习环境

PyTorch 的安装比普通包复杂，因为它有 CPU、CUDA、ROCm、XPU 等不同构建，并且很多 wheel 放在 PyTorch 专用 index 上。

### CPU 版本

在当前虚拟环境中安装 CPU 版本：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

项目模式中也可以先：

```bash
uv add torch torchvision torchaudio
```

如果需要固定 CPU index，可以在 `pyproject.toml` 中配置：

```toml
[project]
name = "dl-study"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "torchvision",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu" },
]
torchvision = [
    { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

### CUDA 版本

CUDA 版本需要结合机器驱动、CUDA runtime 和 PyTorch 支持情况选择。示例：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

项目模式中可以通过 `tool.uv.sources` 指定 PyTorch index：

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

uv 也提供了 PyTorch 后端自动选择能力：

```bash
uv add torch torchvision --torch-backend=auto
```

不过深度学习环境牵涉 GPU 驱动和平台差异，学习初期可以先用 CPU 版本跑通概念，再根据设备情况切换 CUDA 版本。

## 工具运行：uvx 与 uv tool

有些 Python 包主要是命令行工具，例如 `ruff`、`black`、`httpie`、`jupyter`。如果只是临时运行，不一定要安装进项目。

### 临时运行工具

`uvx` 是 `uv tool run` 的别名：

```bash
uvx ruff check .
uvx black --version
uvx pycowsay "hello uv"
```

这类似 `pipx run`：uv 会创建临时隔离环境运行工具。

### 安装工具

如果某个工具经常使用，可以安装成全局命令：

```bash
uv tool install ruff
ruff --version
```

查看已安装工具：

```bash
uv tool list
```

卸载工具：

```bash
uv tool uninstall ruff
```

## 从其他方式迁移到 uv

### 从 venv + pip 迁移

原始方式：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

uv 方式：

```bash
uv venv
uv pip install -r requirements.txt
uv run python main.py
```

如果想转成项目模式：

```bash
uv init
uv add -r requirements.txt
uv sync
```

### 从 Conda 迁移

Conda 原来可能是：

```bash
conda create -n ml python=3.11
conda activate ml
conda install numpy pandas scikit-learn matplotlib
```

uv 项目模式可以写成：

```bash
uv init ml
cd ml
uv python pin 3.11
uv add numpy pandas scikit-learn matplotlib
uv run python main.py
```

如果原来有 `environment.yml`，需要手动拆分：

```yaml
name: ml
dependencies:
  - python=3.11
  - numpy
  - pandas
  - pip
  - pip:
      - scikit-learn
      - matplotlib
```

可以迁移为：

```bash
uv init ml
cd ml
uv python pin 3.11
uv add numpy pandas scikit-learn matplotlib
```

Conda 的优势是能管理非 Python 二进制依赖，例如某些系统库、CUDA 工具链、C/C++ 编译依赖。uv 更偏 Python 包和项目环境管理。如果一个项目强依赖 Conda channel 中的二进制包，迁移时要谨慎。

### 从 Poetry 迁移

Poetry 项目通常已经有 `pyproject.toml`：

```bash
poetry install
poetry add requests
poetry run python main.py
```

uv 中对应：

```bash
uv sync
uv add requests
uv run python main.py
```

两者都使用 `pyproject.toml`，但锁文件不同：

```text
Poetry: poetry.lock
uv:     uv.lock
```

迁移时不要直接手动改锁文件，应该让 uv 重新解析。

### 从 pipx 迁移

pipx 常用于安装命令行工具：

```bash
pipx install ruff
pipx run black --version
```

uv 对应：

```bash
uv tool install ruff
uvx black --version
```

## 常用命令

### 项目管理

```bash
uv init project-name
uv init
uv add requests
uv add --dev pytest ipykernel
uv remove requests
uv run python main.py
uv run pytest
uv lock
uv lock --check
uv sync
uv sync --inexact
```

### 虚拟环境和 pip 接口

```bash
uv venv
uv venv --python 3.11
uv pip install numpy pandas
uv pip install -r requirements.txt
uv pip sync requirements.txt
uv pip list
uv pip show numpy
uv pip freeze
uv pip compile requirements.in -o requirements.txt
```

### Python 版本

```bash
uv python install 3.11
uv python list
uv python pin 3.11
uv run --python 3.11 python --version
```

### 工具

```bash
uvx ruff check .
uv tool run ruff check .
uv tool install ruff
uv tool list
uv tool uninstall ruff
```

### Jupyter

```bash
uv add --dev ipykernel
uv run --with jupyter jupyter lab
uv run ipython kernel install --user --name=project
```

Windows PowerShell 下给 kernel 指定项目环境：

```powershell
uv run ipython kernel install --user --env VIRTUAL_ENV "$PWD\.venv" --name=project
```

## Python 环境管理方式对比

### 总览

| 工具 | 主要解决什么 | 优点 | 局限 |
| --- | --- | --- | --- |
| `venv` | 创建虚拟环境 | Python 自带、简单稳定 | 不负责依赖锁定和 Python 版本安装 |
| `pip` | 安装 Python 包 | 标准工具、生态最广 | 默认不做环境隔离，也不做完整锁定 |
| `pip-tools` | 从 `.in` 生成锁定的 `requirements.txt` | 适合传统 pip 工作流 | 需要搭配 venv/pip 使用 |
| `pipx` | 隔离安装命令行工具 | 适合 black、ruff、httpie 等工具 | 不适合项目依赖管理 |
| `pyenv` | 管理 Python 解释器版本 | 多版本 Python 切换清晰 | 不管理项目依赖 |
| `conda` | 环境 + 包 + 非 Python 依赖 | 科学计算生态成熟，能装很多二进制依赖 | 环境较重，锁定和跨平台复现不总是轻量 |
| `mamba` | 更快的 conda 替代实现 | 解析和安装更快 | 仍属于 conda 生态 |
| `poetry` | Python 项目依赖和打包 | `pyproject.toml` 工作流成熟 | 速度和工具整合度不如 uv 简洁 |
| `pipenv` | Pipfile 项目环境 | 曾经流行，概念完整 | 现在新项目使用率下降 |
| `uv` | Python 项目、环境、依赖、工具和 Python 版本 | 快、统一、支持 lockfile、适合现代项目 | 对非 Python 系统依赖不如 conda |

### 选型建议

| 场景 | 推荐 |
| --- | --- |
| 受限环境中不能使用 Conda，但可以使用 Python | uv |
| 新建 Python 学习项目 | uv 项目模式 |
| 只想快速给当前目录创建虚拟环境 | `uv venv` |
| 复现只有 `requirements.txt` 的老项目 | `uv venv` + `uv pip install -r requirements.txt` |
| 需要严格同步 requirements | `uv pip sync requirements.txt` |
| 需要 Jupyter Notebook 学习 ML | uv + `ipykernel` |
| 需要安装命令行工具 | `uvx` 或 `uv tool install` |
| 需要大量非 Python 二进制依赖 | conda / mamba |
| 需要管理系统级 Python 多版本但不管包 | pyenv |

### uv 与 Conda

| 维度 | uv | Conda |
| --- | --- | --- |
| 环境位置 | 通常是项目内 `.venv` | 通常是集中环境目录 |
| 依赖来源 | PyPI、包 index、Git、路径等 | Conda channel，也可混用 pip |
| Python 版本 | 可安装和 pin Python | 创建环境时指定 Python |
| 锁文件 | `uv.lock` | 通常使用 `environment.yml`，严格锁定需额外方案 |
| 非 Python 依赖 | 不擅长 | 擅长 |
| 速度 | 通常很快 | Conda 较慢，mamba 较快 |
| 适合 | Web、脚本、ML 学习、现代 Python 项目 | 科学计算、大量二进制依赖、复杂 CUDA/系统库 |

对文档型学习项目来说，如果只是 `numpy`、`pandas`、`sklearn`、`matplotlib`、`jupyter`，uv 很合适；如果课程环境要求特定 CUDA、GDAL、OpenCV 系统库、C++ 工具链，Conda 仍然可能更省心。

### uv 与 Poetry

| 维度 | uv | Poetry |
| --- | --- | --- |
| 项目文件 | `pyproject.toml` | `pyproject.toml` |
| 锁文件 | `uv.lock` | `poetry.lock` |
| 运行命令 | `uv run` | `poetry run` |
| 添加依赖 | `uv add` | `poetry add` |
| 同步环境 | `uv sync` | `poetry install` |
| Python 版本安装 | 支持 | 通常依赖外部工具 |
| 工具运行 | `uvx` / `uv tool` | 不是主要功能 |

如果已经是 Poetry 项目，可以继续用 Poetry；如果是新项目，uv 的工具链更统一。

### uv 与 pip + venv

| 维度 | pip + venv | uv |
| --- | --- | --- |
| 创建环境 | `python -m venv .venv` | `uv venv` |
| 安装依赖 | `pip install` | `uv pip install` 或 `uv add` |
| 运行命令 | 先激活再运行 | `uv run` 可自动使用项目环境 |
| 锁定依赖 | 需要 pip-tools 等额外工具 | `uv.lock` |
| Python 版本 | 依赖系统已有 Python | 可 `uv python install` |
| 项目元数据 | 手动维护 | 使用 `pyproject.toml` |

`pip + venv` 是基础能力；uv 是把这些基础能力工程化整合起来。

## 建议工作流

### 学习 ML 的项目工作流

```bash
uv init ml-study
cd ml-study
uv python pin 3.11
uv add numpy pandas matplotlib seaborn scikit-learn
uv add --dev ipykernel jupyterlab
uv sync
uv run python -c "import numpy, pandas, sklearn; print('ok')"
```

然后在 VS Code 中选择：

```text
.venv\Scripts\python.exe
```

或者启动 Jupyter：

```bash
uv run --with jupyter jupyter lab
```

### 克隆 GitHub 项目后的工作流

如果项目已经有 `pyproject.toml` 和 `uv.lock`：

```bash
git clone <repo-url>
cd <repo>
uv sync
uv run python main.py
```

如果项目只有 `requirements.txt`：

```bash
git clone <repo-url>
cd <repo>
uv venv
uv pip install -r requirements.txt
uv run python main.py
```

如果想把它转成 uv 项目：

```bash
uv init
uv add -r requirements.txt
uv sync
```

### 临时跑一个 Python 文件

```bash
uv run script.py
```

如果脚本需要依赖，可以先在项目中添加：

```bash
uv add requests
uv run script.py
```

或者使用内联依赖脚本：

```bash
uv add --script script.py requests
uv run script.py
```

## 参考资料

- [uv 官方文档](https://docs.astral.sh/uv/)
- [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Working on projects](https://docs.astral.sh/uv/guides/projects/)
- [Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [The pip interface](https://docs.astral.sh/uv/pip/)
- [Using Python environments](https://docs.astral.sh/uv/pip/environments/)
- [Using uv with Jupyter](https://docs.astral.sh/uv/guides/integration/jupyter/)
- [Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
