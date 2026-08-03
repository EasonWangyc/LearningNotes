# Docker 容器化

## 内容概览

Docker 是一种以容器为核心的应用打包与运行工具。它把应用程序、运行时、系统库和配置组织成可分发的镜像，再以容器的形式启动。

本笔记按照“概念 -> 命令 -> 配置 -> 实践”的顺序整理：

- 理解镜像、容器、层、仓库和运行时
- 掌握容器生命周期、网络和数据持久化
- 使用 Dockerfile 构建可复现的运行环境
- 使用 Docker Compose 管理多个服务
- 了解 GPU 容器和 NVIDIA Container Toolkit
- 使用 Docker 运行大模型推理服务与训练任务
- 处理模型权重、缓存、共享内存和 GPU 资源
- 认识构建缓存、安全、监控与问题定位方法

---

## 1. Docker 的核心概念

### 1.1 Docker 解决什么问题

一个 Python 或深度学习程序通常不只有源代码，还依赖：

- Python 版本
- 第三方库及其精确版本
- 操作系统动态库
- 编译器和系统工具
- CUDA、cuDNN、NCCL 等 GPU 软件栈
- 环境变量和启动参数
- 模型权重、词表和其他资源文件

如果只分发源代码，接收者还需要手动搭建这些依赖，容易出现“版本相同但行为不同”的情况。

Docker 的基本思路是：

```text
Dockerfile
    |
    |  docker build
    v
镜像 Image
    |
    |  docker run
    v
容器 Container
    |
    |  挂载卷、网络、环境变量、GPU
    v
可运行的服务或任务
```

Docker 不会自动解决所有兼容性问题，但它可以把应用层的运行环境固定下来，并把启动方式变成明确的命令和配置。

### 1.2 镜像 Image

镜像是一个只读的、分层保存的文件系统快照，同时包含启动容器所需的元数据。

镜像通常包含：

- 基础操作系统用户空间
- 运行时，例如 Python 或 Node.js
- 系统依赖
- 应用代码
- 默认启动命令
- 默认环境变量
- 工作目录和用户信息

镜像本身不是正在运行的进程。执行 docker run 后，Docker 才会根据镜像创建容器。

查看本地镜像：

```bash
docker image ls
docker image inspect python:3.12-slim
```

镜像名称通常由以下部分组成：

```text
[registry/]repository[:tag]
```

例如：

```text
docker.io/library/python:3.12-slim
ghcr.io/example/inference-server:1.0.0
```

其中：

- registry：镜像仓库地址，省略时通常使用 Docker Hub
- repository：镜像仓库和名称
- tag：便于人阅读的版本标签
- digest：镜像内容的不可变摘要，形式类似 sha256:...

标签可能被仓库重新指向，digest 更适合严格复现：

```bash
docker pull python:3.12-slim
docker image inspect python:3.12-slim --format '{{index .RepoDigests 0}}'
```

### 1.3 容器 Container

容器是镜像的一个运行实例。它具有：

- 独立的进程空间
- 独立的网络命名空间
- 自己的环境变量
- 自己的可写层
- 可选的卷、绑定挂载、设备和 GPU

容器不是虚拟机。容器内的进程仍然使用宿主机内核，只是在进程、文件、网络和资源视图上受到隔离。

创建容器后，即使主进程退出，容器对象仍可能保留，可以通过 docker start 再次启动：

```bash
docker create --name demo-python python:3.12-slim python --version
docker start demo-python
docker logs demo-python
docker rm demo-python
```

### 1.4 容器、镜像与进程的关系

可以把它们理解为：

```text
镜像：应用运行环境的模板
容器：模板创建出的一个实例
容器主进程：决定容器是否仍然运行的前台进程
```

默认情况下，容器的生命周期与 PID 1 进程相关：

- PID 1 正常退出，容器通常变为 Exited
- PID 1 被杀死，容器通常也会停止
- 后台服务必须以前台进程运行，否则容器会立刻退出

示例：

```bash
docker run --name short-task python:3.12-slim python -c "print('container finished')"
docker ps -a --filter name=short-task
docker rm short-task
```

### 1.5 层 Layer

Dockerfile 中的大多数指令会产生镜像层。层具有以下特点：

- 只读
- 可以被后续镜像复用
- 根据指令和输入内容参与缓存判断
- 多个镜像可以共享相同的基础层

容器启动时，Docker 会在镜像只读层之上增加一个可写层：

```text
容器可写层              运行期间产生的临时文件
-----------------------------------------------
应用层                  当前镜像新增的内容
基础层                  Python、系统库等
```

容器可写层随容器删除而消失，因此数据库、模型权重、用户上传文件等重要数据不应只放在这里。

### 1.6 Registry、Repository、Tag 与 Digest

镜像仓库用于保存和分发镜像。常见操作包括：

```bash
docker login
docker pull nginx:1.27
docker tag local-image:latest registry.example.com/demo/app:1.0.0
docker push registry.example.com/demo/app:1.0.0
```

建议：

- 发布镜像时使用有含义的版本标签
- 重要运行环境同时记录 digest
- 不要把 latest 当成严格的版本号
- 基础镜像、Python 依赖和服务代码都应记录来源

### 1.7 Docker 与虚拟机的区别

| 对比项 | Docker 容器 | 虚拟机 |
| --- | --- | --- |
| 隔离边界 | 进程和内核资源隔离 | 虚拟硬件与完整操作系统 |
| 启动速度 | 通常较快 | 通常较慢 |
| 镜像大小 | 通常较小 | 通常包含完整系统 |
| 内核 | 使用宿主机内核 | 每台虚拟机有自己的内核 |
| 隔离强度 | 依赖内核和配置 | 通常更强 |
| 适合场景 | 服务、任务、构建环境 | 不同内核、强隔离、完整系统 |
| GPU 使用 | 通过设备映射和运行时接入 | 需要虚拟化或直通方案 |

容器并不等于安全边界。使用 privileged、暴露 Docker socket、共享宿主机网络或 IPC 时，隔离边界会进一步变弱。

---

## 2. 大模型时代 Docker 的作用

大模型系统的运行环境通常比普通 Web 服务更复杂，Docker 的价值主要集中在以下几个方面。

### 2.1 固定复杂的软件栈

一个模型服务可能同时依赖：

- 特定的 Python 版本
- PyTorch 或其他深度学习框架
- CUDA runtime
- GPU 加速库
- tokenizer 和模型服务框架
- 编译扩展
- HTTP API 服务
- 监控与日志组件

Docker 可以把这些用户态依赖放入一个明确的镜像中，减少不同机器之间的环境差异。

需要注意：Docker 主要固定用户空间，不会替代宿主机 GPU 驱动、内核和硬件。GPU 容器能否工作仍然取决于宿主机驱动、容器运行时和容器内 CUDA 用户态库之间的兼容关系。

### 2.2 把模型服务变成可部署的接口

大模型的调用方通常不需要直接了解模型加载代码，而是通过 HTTP 或 OpenAI 兼容接口发送请求：

```text
客户端
   |
   | HTTP / OpenAI-compatible API
   v
容器中的模型服务
   |
   | tokenizer、推理引擎、GPU
   v
模型权重
```

因此可以把：

- 模型服务
- 网关
- 向量数据库
- Redis
- 监控服务

分别放在独立容器中，再用 Compose 管理它们的网络和生命周期。

### 2.3 将镜像、模型权重和运行缓存分开

镜像适合保存代码和运行时，模型权重通常不应频繁写入镜像：

```text
镜像：Python、框架、服务代码、系统库
模型目录：权重、配置、tokenizer
缓存目录：下载缓存、编译缓存、引擎缓存
容器可写层：临时文件
```

这样做的好处：

- 更新代码时不需要重复构建和传输几十 GB 的模型
- 多个容器可以复用相同的模型目录
- 模型版本可以独立切换
- 删除容器不会删除持久化模型和缓存

### 2.4 隔离不同模型服务的资源

一台机器可能需要运行多个模型服务。Docker 可以通过以下方式组织资源：

- 为不同服务分配不同 GPU
- 限制 CPU 和内存
- 限制容器可见的 GPU
- 隔离端口和网络
- 为每个服务使用独立的模型目录
- 使用 Compose 统一管理启动参数

GPU 隔离并不等于 GPU 显存可以自动安全超分配。模型服务仍然要根据显存、上下文长度、并发数和 KV cache 需求进行配置。

### 2.5 固定基准环境

推理性能不仅取决于模型，还会受到以下因素影响：

- GPU 型号和显存
- 驱动版本
- CUDA 和框架版本
- 推理后端
- 精度和量化方式
- batch size
- 输入输出长度
- 并发数
- CPU、内存和共享内存

Docker 可以固定其中的用户态环境，但性能记录仍应包含硬件、驱动、镜像版本、模型版本和完整运行参数。

### 2.6 连接本地、单机和集群环境

同一个镜像可以作为：

- 本地开发环境
- 单机推理容器
- GPU 服务器上的服务
- Kubernetes Pod 的基础镜像
- CI 中的构建和测试环境

Docker 负责打包和进程运行；编排、调度、服务发现、自动扩缩容和跨节点网络通常由更上层的平台负责。

---

## 3. 常用命令

### 3.1 镜像命令

```bash
# 拉取镜像
docker pull python:3.12-slim

# 查看本地镜像
docker image ls

# 查看镜像详细元数据
docker image inspect python:3.12-slim

# 删除镜像
docker image rm python:3.12-slim

# 清理没有被使用的镜像层
docker image prune

# 查看镜像构建历史
docker history python:3.12-slim
```

### 3.2 容器命令

```bash
# 前台运行，主进程退出后命令结束
docker run --name demo-python python:3.12-slim python --version

# 后台运行
docker run -d --name demo-nginx -p 8080:80 nginx:1.27

# 查看运行中的容器
docker ps

# 查看所有容器，包括已退出容器
docker ps -a

# 查看日志
docker logs demo-nginx
docker logs -f demo-nginx

# 查看容器内的进程
docker top demo-nginx

# 进入运行中的容器
docker exec -it demo-nginx sh

# 停止、启动、重启
docker stop demo-nginx
docker start demo-nginx
docker restart demo-nginx

# 删除容器
docker rm demo-nginx

# 强制停止并删除容器
docker rm -f demo-nginx
```

docker run 是一个组合操作，通常会完成：

1. 检查本地是否存在镜像
2. 必要时从仓库拉取镜像
3. 创建容器
4. 配置文件系统、网络、环境变量和挂载
5. 启动容器主进程

### 3.3 环境变量、端口与工作目录

```bash
docker run --rm \
  --name env-demo \
  -e APP_ENV=development \
  -e LOG_LEVEL=info \
  -w /workspace \
  python:3.12-slim \
  python -c "import os; print(os.getcwd(), os.getenv('APP_ENV'))"
```

端口映射的格式为：

```text
宿主机端口:容器端口
```

例如：

```bash
docker run -d --name web -p 127.0.0.1:8080:80 nginx:1.27
```

只绑定到 127.0.0.1 时，服务通常只能从本机访问；绑定到 0.0.0.0 时，可能对外部网卡开放，需要结合防火墙和访问控制判断。

### 3.4 查看容器配置和资源

```bash
docker inspect demo-nginx
docker stats
docker stats demo-nginx
docker port demo-nginx
docker diff demo-nginx
```

### 3.5 系统清理

清理前先确认没有重要数据：

```bash
docker system df

# 只清理停止的容器
docker container prune

# 只清理未使用的网络
docker network prune

# 只清理未使用的卷
docker volume prune

# 清理未使用的对象，范围较大
docker system prune
```

---

## 4. Docker 中的数据与存储

### 4.1 容器可写层为什么不适合保存重要数据

容器启动后会拥有一个可写层。应用写入容器内部的文件通常进入这个可写层，但它具有以下限制：

- 容器删除后数据通常消失
- 不适合多个容器共享
- 容器层可能带来额外的写放大
- 不方便独立备份和迁移

因此应将重要数据放在 volume 或 bind mount 中。Docker 官方对 writable layer、bind mount、volume 和 tmpfs 的区别有详细说明：[Docker storage](https://docs.docker.com/engine/storage)。

### 4.2 Bind mount

Bind mount 把宿主机上的明确路径映射到容器内：

```bash
docker run --rm \
  -v "$PWD/data:/data" \
  python:3.12-slim \
  python -c "from pathlib import Path; print(list(Path('/data').iterdir()))"
```

适合：

- 本地开发时同步源代码
- 直接访问宿主机上的数据
- 需要明确控制宿主机目录的场景

需要注意：

- 容器内的写操作可能直接修改宿主机文件
- 宿主机路径必须存在或行为符合预期
- Windows、Linux 和远程 Docker 环境的路径语法可能不同
- 只读挂载可以降低误写风险

```bash
docker run --rm \
  --mount type=bind,src="$PWD/config",dst=/etc/app-config,readonly \
  alpine:3.20 \
  ls -la /etc/app-config
```

### 4.3 Named volume

Named volume 由 Docker 管理，适合持久化服务数据：

```bash
docker volume create model-cache
docker volume ls
docker volume inspect model-cache

docker run --rm \
  --mount source=model-cache,target=/cache \
  alpine:3.20 \
  sh -c "echo cached-data > /cache/example.txt"

docker run --rm \
  --mount source=model-cache,target=/cache,readonly \
  alpine:3.20 \
  cat /cache/example.txt
```

适合：

- 模型下载缓存
- 数据库数据
- 编译缓存
- 多次运行之间需要保留的服务数据

### 4.4 tmpfs

tmpfs 将数据放在内存中，容器停止后不保留：

```bash
docker run --rm \
  --tmpfs /tmp:size=256m \
  alpine:3.20 \
  df -h /tmp
```

适合：

- 短期临时文件
- 不希望落盘的中间数据
- 对磁盘写入敏感的临时目录

tmpfs 会占用宿主机内存，不能把它当作无限空间。

### 4.5 大模型权重与缓存的目录设计

一个常见的目录划分如下：

```text
/opt/app                  应用代码
/models                   模型权重，只读挂载
/cache/huggingface        Hugging Face 下载缓存
/cache/torch              PyTorch 或编译缓存
/cache/engines            TensorRT 等推理引擎缓存
/tmp                      临时文件
```

示例：

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$PWD/models",dst=/models,readonly \
  --mount source=hf-cache,target=/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  inference-image:1.0 \
  python -m app --model /models/model-a
```

模型权重可以通过只读方式挂载，防止服务进程意外覆盖原始文件。缓存则可以使用可写 volume，这样不同容器启动时能够复用已经下载或编译的内容。

---

## 5. Dockerfile：构建自己的镜像

### 5.1 Dockerfile 的执行过程

一个 Dockerfile 通常包含：

1. 选择基础镜像
2. 设置工作目录
3. 安装依赖
4. 复制代码和配置
5. 声明端口
6. 设置默认启动命令

最小示例：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]
```

构建并运行：

```bash
docker build -t python-demo:1.0 .
docker run --rm python-demo:1.0
```

### 5.2 CMD 与 ENTRYPOINT

CMD 提供默认命令，可以被 docker run 后面的命令覆盖。

ENTRYPOINT 定义更固定的入口，后面的参数会追加到入口命令后。

```dockerfile
ENTRYPOINT ["python", "-m", "app"]
CMD ["--port", "8000"]
```

运行：

```bash
docker run --rm app-image:1.0
docker run --rm app-image:1.0 --port 9000
```

实际使用时优先采用 JSON 数组形式，避免额外 shell 进程和信号转发问题。

### 5.3 CPU Python 服务镜像

下面的例子使用一个简单 HTTP 服务：

```python
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    """返回一个最小的健康检查响应。"""

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
```

对应 Dockerfile：

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY app.py .

EXPOSE 8000
CMD ["python", "app.py"]
```

构建后测试：

```bash
docker build -t health-demo:1.0 .
docker run --rm -d --name health-demo -p 8000:8000 health-demo:1.0
curl http://127.0.0.1:8000/health
docker rm -f health-demo
```

### 5.4 GPU 基础镜像

GPU 镜像需要选择与框架兼容的 CUDA runtime。示意：

```dockerfile
FROM nvidia/cuda:<cuda-runtime-tag>-runtime-ubuntu<ubuntu-version>

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .
CMD ["python3", "-m", "app"]
```

这里的 cuda-runtime-tag 和 ubuntu-version 只是占位符。实际构建前需要根据框架、驱动、GPU 架构和目标推理后端选择已验证的版本。

常见的基础镜像类型：

| 类型 | 内容 | 适合场景 |
| --- | --- | --- |
| base | CUDA 运行时及基础组件 | 需要自行安装上层依赖 |
| runtime | 运行程序所需的 CUDA 用户态库 | 推理和生产运行 |
| devel | 额外包含编译器和头文件 | 编译 CUDA 扩展或自定义算子 |

### 5.5 多阶段构建

编译阶段和运行阶段通常需要不同的依赖。多阶段构建可以只把编译结果复制到最终镜像：

```dockerfile
FROM python:<python-version> AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --prefix=/install -r requirements.txt

COPY src/ src/
RUN python -m compileall src/


FROM python:<python-version>-slim AS runtime

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /build/src ./src

CMD ["python", "-m", "src.app"]
```

在包含 CUDA 编译扩展的场景中，builder 可以使用 devel 镜像，runtime 使用 runtime 镜像，从而减少最终镜像中不需要的编译器和头文件。

### 5.6 .dockerignore

构建上下文会被发送给 Docker daemon。没有 .dockerignore 时，模型权重、日志、缓存和 Git 历史都可能被发送，导致构建慢、上下文过大，甚至把不应进入镜像的文件打包进去。

示例：

```text
.git
.gitignore
.venv
__pycache__
*.pyc
*.log
.cache
data
models
checkpoints
outputs
notebooks
.env
```

大模型场景尤其要排除：

- 几 GB 到几百 GB 的模型权重
- tokenizer 缓存
- 实验输出
- TensorRT engine
- 训练 checkpoint
- 本地密钥和 .env

### 5.7 构建缓存

Docker 会缓存已经执行过的构建步骤。Docker 官方建议合理安排层顺序、缩小构建上下文，并使用 BuildKit cache mount 来复用依赖下载缓存，详见 [Docker build cache optimization](https://docs.docker.com/build/cache/optimize/)。

推荐把变化频率低的依赖安装放在变化频率高的源代码之前：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 依赖文件变化较少，优先单独复制，便于复用这一层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源代码变化较频繁，放在依赖安装之后
COPY src/ ./src/

CMD ["python", "-m", "src.app"]
```

使用 BuildKit 缓存挂载：

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "app"]
```

对于大模型镜像，缓存优化可以减少以下内容的重复下载：

- Python wheel
- 编译依赖
- CUDA 扩展的构建中间文件

模型权重下载缓存通常更适合放在 volume，而不是构建过程中写入镜像。

### 5.8 固定依赖与构建参数

仅仅固定基础镜像标签还不够，还应固定：

- Python 包版本
- CUDA 和框架组合
- 服务框架版本
- tokenizer 版本
- 模型版本或 commit
- 编译选项
- 构建参数

涉及严格复现时，建议在构建记录中保存：

```text
镜像 tag
镜像 digest
Dockerfile
依赖锁定文件
模型名称与 revision
宿主机 GPU 型号
宿主机驱动版本
容器启动命令
```

---

## 6. Docker Compose

### 6.1 Compose 解决什么问题

当一个系统包含多个容器时，手动维护多个 docker run 命令会比较繁琐。Compose 使用一个 YAML 文件描述：

- 服务
- 镜像或构建方式
- 端口
- 环境变量
- 卷
- 网络
- 健康检查
- 资源限制
- 服务依赖关系

### 6.2 一个 CPU 多服务示例

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      MODEL_ENDPOINT: http://model-server:8000
    ports:
      - "8080:8080"
    depends_on:
      model-server:
        condition: service_healthy

  model-server:
    image: model-server:<pinned-tag>
    command:
      - "--model"
      - "/models/<model-id>"
    volumes:
      - type: bind
        source: ./models
        target: /models
        read_only: true
      - type: volume
        source: model-cache
        target: /cache
    environment:
      MODEL_CACHE_DIR: /cache
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 12

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  model-cache:
  redis-data:
```

这里的 curl 必须存在于模型服务镜像中。如果镜像没有 curl，可以改用 Python、服务自带的健康检查命令，或者在镜像中安装检查工具。

常用命令：

```bash
docker compose config
docker compose pull
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f model-server
docker compose exec api sh
docker compose down
```

docker compose down 默认删除容器和网络，通常不会删除命名卷；如果加上 -v，会同时删除 Compose 管理的卷，应谨慎使用。

### 6.3 Compose 中的网络

Compose 默认会为同一个应用创建网络。服务之间可以直接使用服务名通信：

```text
api -> http://model-server:8000
api -> redis:6379
```

容器之间通信时使用容器端口，不需要使用宿主机映射端口。ports 用于宿主机或外部访问，expose 主要用于记录和服务间可见性。

### 6.4 健康检查与启动顺序

depends_on 只能描述依赖关系，不能保证服务已经能够处理业务请求。健康检查可以让依赖服务在满足条件后再被认为可用，但应用本身仍应实现：

- 启动重试
- 请求超时
- 连接断开后的重连
- 模型加载失败后的明确错误
- 优雅退出

大模型服务的“进程已启动”和“模型已加载完成”是两个不同状态，健康检查最好检查实际推理接口或模型加载状态。

---

## 7. GPU 容器

### 7.1 GPU 容器的组成

GPU 容器通常涉及四个层次：

```text
GPU 硬件
    |
宿主机 NVIDIA 驱动
    |
NVIDIA Container Toolkit / Docker runtime
    |
容器内 CUDA、框架、推理后端
```

NVIDIA Container Toolkit 负责把宿主机 GPU 能力以容器可用的方式注入运行时。当前官方文档以 nvidia-container-toolkit 和 nvidia-ctk 为核心，详见 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)。

容器不能自行替代宿主机驱动。容器内通常携带 CUDA 用户态库，宿主机提供驱动和设备接口。

### 7.2 配置 NVIDIA Container Toolkit

以下命令适用于已经安装 Docker 和 NVIDIA 驱动的 Linux 主机：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

实际安装步骤和发行版差异应以 [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 为准。

### 7.3 验证 GPU 是否能被容器访问

```bash
docker run --rm --gpus all \
  nvidia/cuda:<cuda-runtime-tag>-base-ubuntu<ubuntu-version> \
  nvidia-smi
```

这个测试只验证：

- Docker 能创建 GPU 容器
- NVIDIA runtime 能挂载 GPU 设备
- 容器能调用 nvidia-smi

它不能证明：

- PyTorch 一定可用
- 模型一定能加载
- CUDA kernel 一定兼容
- 推理性能满足要求

### 7.4 选择 GPU

使用所有 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi
```

只使用编号为 0 的 GPU：

```bash
docker run --rm --gpus '"device=0"' nvidia/cuda:<tag> nvidia-smi
```

使用多个指定 GPU：

```bash
docker run --rm --gpus '"device=0,2"' nvidia/cuda:<tag> nvidia-smi
```

容器内看到的 GPU 编号可能与宿主机逻辑编号不同。应用程序还可能通过 CUDA_VISIBLE_DEVICES 进一步改变可见设备编号。

### 7.5 Compose 使用 GPU

Docker Compose 可以通过设备预留配置声明 GPU。Docker 官方示例要求 capabilities 包含 gpu，并在 count 与 device_ids 之间选择一种方式，详见 [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)。

使用指定数量的 GPU：

```yaml
services:
  inference:
    image: inference-image:<pinned-tag>
    command: ["python", "-m", "app"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

使用指定设备：

```yaml
services:
  inference:
    image: inference-image:<pinned-tag>
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

count 和 device_ids 不应同时设置。

### 7.6 GPU 兼容性检查顺序

当 GPU 容器无法运行时，可以按以下顺序检查：

```text
宿主机能否执行 nvidia-smi
    |
Docker 是否安装且能启动容器
    |
NVIDIA Container Toolkit 是否配置到 Docker
    |
docker run --gpus all 是否能看到 GPU
    |
容器内 CUDA 与框架是否匹配
    |
模型是否能加载
    |
性能和显存是否满足要求
```

不要只看到 nvidia-smi 输出就认为完整推理链路已经验证成功。

---

## 8. 大模型推理服务

### 8.1 推理服务需要管理哪些资源

一个大模型推理容器通常同时管理：

| 资源 | 作用 |
| --- | --- |
| 模型权重 | 参数、配置、tokenizer |
| GPU 显存 | 权重、激活值、KV cache |
| CPU 内存 | 权重加载、预处理、请求队列 |
| 共享内存 | 多进程或框架间通信 |
| 网络端口 | API 请求和健康检查 |
| 模型缓存 | 下载、编译和引擎复用 |
| 日志与指标 | 观察吞吐、延迟和错误 |

模型服务的主要瓶颈可能来自显存、共享内存、磁盘加载、CPU 预处理或请求排队，而不一定只是 GPU 计算。

### 8.2 使用 vLLM 容器

vLLM 官方提供了 Docker 部署说明和镜像。下面是一个通用命令模板，版本标签和模型 ID 需要替换为实际验证过的值：

```bash
docker run --runtime nvidia --gpus all \
  --name vllm \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:<pinned-tag> \
  --model <model-id>
```

官方示例和参数说明见 [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/)。

命令中的关键部分：

- --gpus all：把 GPU 暴露给容器
- -v hf-cache：持久化模型下载缓存
- HF_TOKEN：访问需要权限的模型仓库时使用
- -p 8000:8000：映射 OpenAI 兼容 API 端口
- --ipc=host：让容器使用宿主机 IPC 命名空间
- --model：指定模型名称、路径或本地目录

--ipc=host 会放宽容器的 IPC 隔离。若不希望使用 host IPC，可以根据框架说明设置足够大的 --shm-size，例如：

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  vllm/vllm-openai:<pinned-tag> \
  --model <model-id>
```

共享内存参数不是越大越好，应根据多进程数量、数据传输方式和实际错误调整。

### 8.3 调用 OpenAI 兼容接口

容器启动后，可以使用 HTTP 请求验证服务：

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Explain container isolation in one paragraph."}],"max_tokens":64}'
```

如果服务启用了鉴权，还需要在请求中添加对应的 Authorization 头。

### 8.4 Compose 管理模型服务

```yaml
services:
  llm:
    image: vllm/vllm-openai:<pinned-tag>
    command:
      - "--model"
      - "/models/<model-id>"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
    ports:
      - "8000:8000"
    volumes:
      - type: bind
        source: ./models
        target: /models
        read_only: true
      - type: volume
        source: hf-cache
        target: /root/.cache/huggingface
    environment:
      HF_HOME: /root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: "8gb"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 15s
      timeout: 5s
      retries: 20

volumes:
  hf-cache:
```

实际使用时还需要考虑：

- API 鉴权
- 请求超时和最大输入长度
- 并发请求上限
- 日志脱敏
- 模型目录权限
- 模型服务的优雅停止
- 健康检查是否真的等待模型加载完成
- 端口是否只绑定到需要的网卡

### 8.5 模型权重不一定要放进镜像

把模型权重写入镜像有时很方便，但需要权衡：

| 方式 | 优点 | 代价 |
| --- | --- | --- |
| 写入镜像 | 启动时不依赖外部下载 | 镜像巨大，更新和分发慢 |
| Bind mount | 权重与镜像分离，容易切换 | 依赖宿主机目录 |
| Named volume | Docker 管理，适合缓存 | 需要单独初始化和备份 |
| 启动时下载 | 部署简单 | 启动受网络影响，需管理凭证 |
| 预置本地目录 | 启动稳定 | 需要明确目录和权限 |

学习和验证阶段通常适合“镜像保存代码与运行时，模型通过挂载目录或缓存 volume 提供”。

### 8.6 区分不同类型的缓存

大模型系统中常见的缓存不是同一种东西：

```text
Docker build cache
    构建镜像时复用 Dockerfile 层

Python/package cache
    复用 wheel 和包下载结果

Model download cache
    复用模型权重、配置和 tokenizer

Engine cache
    复用 TensorRT 或其他后端生成的引擎

KV cache
    推理过程中保存上下文的键值状态，通常位于 GPU 显存
```

它们的生命周期和存储位置不同。KV cache 不是 Docker volume 能够替代的持久化缓存，它属于推理运行时的内存管理。

---

## 9. 大模型训练与分布式运行

### 9.1 单机多 GPU 训练

单机多 GPU 训练通常需要：

- GPU 设备访问
- 足够的共享内存
- 进程间通信
- 正确的 NCCL 配置
- 容器内的训练代码和数据
- checkpoint 的持久化目录

示意命令：

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/src:/workspace/src" \
  -v "$PWD/data:/workspace/data:ro" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  training-image:<pinned-tag> \
  torchrun --nproc_per_node=4 /workspace/src/train.py \
    --data /workspace/data \
    --output /workspace/checkpoints
```

参数含义：

- --nproc_per_node=4：启动 4 个训练进程
- --gpus all：让容器看到所有 GPU
- --ipc=host：为进程间通信提供更大的共享空间
- --ulimit memlock=-1：放宽锁页内存限制，具体是否需要取决于通信库
- checkpoint 目录挂载到宿主机，避免容器删除后丢失训练结果

### 9.2 多节点训练不是单个 Docker 命令就能完成

跨节点训练还需要解决：

- 节点发现
- 主节点地址和端口
- rank 与 world size
- 节点间网络连通
- NCCL 或其他通信库
- 数据共享和 checkpoint 协调
- 节点故障与任务恢复

Docker 主要提供每个节点上的一致运行环境。跨节点编排通常需要作业调度器、Kubernetes、专门的训练平台或其他分布式运行系统。

### 9.3 Host network 的取舍

某些分布式通信场景会使用：

```bash
docker run --rm --network=host --gpus all training-image:<pinned-tag>
```

--network=host 可以减少端口映射和网络命名空间带来的复杂度，但它会降低网络隔离。只有在明确理解端口暴露和访问边界后才使用。

### 9.4 数据和 checkpoint 的持久化

训练容器通常应把以下目录挂载到外部存储：

```text
数据集             只读挂载或专用数据卷
checkpoint         可写持久化目录
日志               可写日志目录
TensorBoard 文件   可写输出目录
编译和模型缓存      可写缓存卷
```

不要把唯一的 checkpoint 放在容器可写层中，也不要让训练进程直接覆盖不可恢复的原始数据。

---

## 10. 构建效率与可复现性

### 10.1 让构建上下文尽可能小

构建上下文越大：

- 上传给 Docker daemon 的时间越长
- 缓存失效判断越复杂
- 无意中包含敏感文件的风险越高

.dockerignore 应排除模型、数据、缓存、日志、虚拟环境和版本控制目录。

### 10.2 优先安装稳定依赖

推荐的 Dockerfile 顺序：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv \
    && uv sync --frozen --no-dev

COPY src/ ./src/

CMD ["uv", "run", "python", "-m", "src.app"]
```

如果使用 requirements.txt、Poetry 或其他包管理方式，原则相同：先复制锁定文件，安装依赖后再复制源代码。

### 10.3 标签与 digest

开发阶段可以使用语义标签：

```text
inference-server:0.3.0
```

需要严格复现时同时记录 digest：

```bash
docker image inspect inference-server:0.3.0 \
  --format '{{json .RepoDigests}}'
```

基础镜像和第三方服务镜像也应避免只依赖会变化的 latest。

### 10.4 记录环境信息

可以在容器启动时输出关键版本：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
nvidia-smi
docker version
```

对于性能实验，还应记录：

- 输入长度和输出长度
- batch size 和并发数
- 精度、量化和编译选项
- 模型 revision
- GPU 显存占用
- 首 token 延迟和持续生成速度
- 是否使用缓存 volume、engine cache 和 warmup

---

## 11. 安全与权限

### 11.1 不要把密钥写入镜像

以下写法会让密钥进入镜像历史或构建缓存，不应使用：

```dockerfile
# 不要这样做
ENV HF_TOKEN=secret-value
ARG API_KEY=secret-value
RUN echo "secret-value"
```

也不要把 .env、私钥、访问令牌和云平台配置复制进镜像。

运行时可以通过环境变量或 env 文件注入：

```bash
docker run --rm \
  --env-file .env \
  inference-image:<pinned-tag>
```

更适合服务编排的方式是 Compose secrets。Docker Compose 会把 secret 作为文件挂载到 /run/secrets/<secret-name>，并且只有声明了该 secret 的服务可以访问，详见 [Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)。

```yaml
services:
  llm:
    image: inference-image:<pinned-tag>
    secrets:
      - hf_token
    environment:
      HF_TOKEN_FILE: /run/secrets/hf_token

secrets:
  hf_token:
    file: ./secrets/hf_token.txt
```

应用需要从 HF_TOKEN_FILE 指向的文件读取令牌，而不是把令牌写入镜像。

### 11.2 使用非 root 用户

示例：

```dockerfile
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
CMD ["python", "-m", "app"]
```

使用非 root 用户可以降低应用进程被利用后对容器文件系统造成破坏的影响，但它不能替代主机安全配置。

### 11.3 只读文件系统与最小权限

可以根据应用特性使用：

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp:size=256m \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  app-image:<pinned-tag>
```

如果应用确实需要写入缓存，应显式挂载可写的缓存目录，而不是让整个根文件系统可写。

### 11.4 谨慎使用高权限选项

以下选项会显著改变隔离边界：

- privileged
- network=host
- ipc=host
- 挂载 /var/run/docker.sock
- 映射宿主机根目录
- 直接暴露设备文件
- 使用过宽的 Linux capabilities

GPU 训练和推理有时需要 --gpus 或共享内存配置，但不代表必须使用 privileged。

### 11.5 镜像安全扫描

如果本地安装了 Docker Scout，可以检查镜像中的依赖漏洞：

```bash
docker scout cves inference-image:<pinned-tag>
```

扫描结果需要结合实际情况处理：

- 更新基础镜像
- 升级存在漏洞的系统包
- 升级 Python 依赖
- 删除不需要的编译工具
- 评估漏洞是否会影响实际运行路径

---

## 12. 监控与性能观察

### 12.1 容器资源

```bash
docker stats
docker stats llm-service
docker inspect llm-service
```

docker stats 可以观察 CPU、内存、网络和块设备 I/O。它不能完整反映 GPU 利用率和 GPU 显存。

### 12.2 GPU 资源

在宿主机或 GPU 容器中：

```bash
nvidia-smi
nvidia-smi dmon
```

在 Windows PowerShell 中，watch 不一定存在，可以重复执行 nvidia-smi，或者使用系统任务调度和监控工具。

### 12.3 大模型服务的关键指标

大模型推理常见指标包括：

| 指标 | 含义 |
| --- | --- |
| TTFT | 从请求到生成第一个 token 的时间 |
| Inter-token latency | 连续 token 之间的平均间隔 |
| Throughput | 单位时间生成的 token 数或完成请求数 |
| Queue time | 请求在服务队列中等待的时间 |
| GPU memory | 权重、激活值和 KV cache 占用的显存 |
| Request concurrency | 同时处理的请求数量 |
| Error rate | 请求失败、超时或取消的比例 |
| Model load time | 服务启动到模型可用所需时间 |

只观察平均吞吐量可能掩盖排队、长尾延迟和显存不足问题。至少应区分冷启动、warmup 和稳定运行阶段。

### 12.4 常见测量误区

- 把容器启动时间当作模型加载时间
- 未等待 warmup 就记录首个请求
- 不记录输入输出长度
- 不区分单请求延迟和并发吞吐
- 只观察 GPU 利用率，不观察显存和 CPU
- 只测量服务端，不测量客户端网络和排队
- 更新模型或镜像后仍沿用旧结论
- 把缓存命中和缓存未命中的结果混在一起

---

## 13. 常见问题定位

### 13.1 容器看不到 GPU

逐项检查：

```bash
nvidia-smi
docker info
docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi
docker inspect <container-name>
```

常见原因：

- 宿主机驱动没有正常工作
- NVIDIA Container Toolkit 未安装或未配置
- Docker daemon 未重启
- --gpus 参数缺失
- Compose 未声明 GPU reservation
- 容器镜像和运行时架构不匹配

### 13.2 模型路径或权重加载失败

检查：

```bash
docker inspect <container-name>
docker exec -it <container-name> sh
ls -lah /models
ls -lah /root/.cache/huggingface
```

常见原因：

- 宿主机路径写错
- 容器内路径与启动参数不一致
- 使用了只读挂载但程序需要写缓存
- 文件权限不足
- 模型文件不完整
- 模型格式与推理后端不匹配
- 访问私有模型时没有提供有效凭证

### 13.3 GPU 显存不足

可能原因：

- 模型权重本身超出显存
- 上下文长度过大
- batch size 或并发过高
- KV cache 占用过多
- 精度或量化配置不合适
- 多个服务同时占用 GPU
- 旧容器仍在运行

检查：

```bash
nvidia-smi
docker ps
docker stats
```

调整方向：

- 降低并发和 batch size
- 降低最大上下文长度
- 选择合适的精度或量化方式
- 使用张量并行或多 GPU
- 确认是否有未使用的进程占用显存
- 为不同服务明确分配可见 GPU

### 13.4 共享内存不足

常见表现：

- 多进程数据加载失败
- NCCL 或进程间通信报错
- PyTorch DataLoader 异常
- 服务启动后在高并发下崩溃

可以尝试：

```bash
docker run --rm \
  --shm-size=8g \
  --gpus all \
  inference-image:<pinned-tag>
```

Compose 中可使用：

```yaml
services:
  inference:
    image: inference-image:<pinned-tag>
    shm_size: "8gb"
```

需要根据进程数、数据加载方式和通信模式实际调整。

### 13.5 容器启动后立即退出

检查：

```bash
docker ps -a
docker logs <container-name>
docker inspect <container-name> --format '{{json .State}}'
```

常见原因：

- 主进程执行完毕
- 启动命令路径错误
- 环境变量缺失
- 配置文件没有挂载
- 权重加载失败
- 端口或设备初始化失败

### 13.6 网络请求无法访问

检查：

- 服务是否监听 0.0.0.0 而不是 127.0.0.1
- 是否正确映射宿主机端口
- 容器之间是否使用服务名通信
- 防火墙是否允许访问
- Compose 网络是否创建成功
- 服务是否已经通过健康检查

命令：

```bash
docker port <container-name>
docker network ls
docker network inspect <network-name>
```

---

## 14. 一个完整的学习流程

下面的流程适合逐步验证 Docker 能力。

### 14.1 运行一个最小容器

```bash
docker run --rm python:3.12-slim python -c "print('hello from container')"
```

目标：理解镜像拉取、容器创建和主进程退出。

### 14.2 构建一个自己的镜像

```bash
docker build -t hello-app:1.0 .
docker run --rm hello-app:1.0
```

目标：理解 Dockerfile、构建上下文和镜像层。

### 14.3 加入目录挂载和环境变量

```bash
docker run --rm \
  -e APP_ENV=dev \
  -v "$PWD/data:/app/data" \
  hello-app:1.0
```

目标：理解镜像内容和运行时配置的边界。

### 14.4 使用 Compose 运行多个服务

```bash
docker compose config
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down
```

目标：理解服务名、网络、卷和健康检查。

### 14.5 验证 GPU 容器

```bash
docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi
```

目标：区分 Docker 基础能力和 GPU runtime 能力。

### 14.6 运行一个模型服务

启动模型服务后依次验证：

```text
容器是否处于 running
    |
模型目录是否可读
    |
GPU 是否可见
    |
健康检查是否通过
    |
模型列表接口是否返回
    |
最小推理请求是否成功
    |
并发、延迟和显存是否可观察
```

不要把“容器启动成功”直接等同于“模型服务可用”。

---

## 15. 学习顺序

推荐顺序：

1. 镜像、容器和常用命令
2. Dockerfile 和镜像层
3. bind mount、named volume 和网络
4. Compose、健康检查和服务依赖
5. .dockerignore、构建缓存和依赖锁定
6. NVIDIA Container Toolkit 和 GPU 访问
7. 模型权重、Hugging Face cache 和共享内存
8. vLLM 等模型服务的容器化运行
9. 单机多 GPU 训练与通信参数
10. 安全、监控、性能和问题定位

每学完一个部分，最好保留以下信息：

- 使用的镜像和版本
- 完整启动命令
- 宿主机与容器内的路径
- 关键环境变量
- 预期输出
- 失败时的日志和定位过程

---

## 16. 参考资料

- [Docker Engine documentation](https://docs.docker.com/engine/)
- [Docker storage](https://docs.docker.com/engine/storage)
- [Docker volumes](https://docs.docker.com/engine/storage/volumes/)
- [Docker build cache optimization](https://docs.docker.com/build/cache/optimize/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/)