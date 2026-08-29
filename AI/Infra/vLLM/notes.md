# vLLM

> 大语言模型高效推理框架学习笔记

## 内容概览

- vLLM 核心机制：PagedAttention
- 快速部署 LLM 推理服务（OpenAI-compatible API）
- 支持模型：LLaMA、Qwen、Mistral 等
- 量化（AWQ、GPTQ）与多 GPU 部署
- 与 LangChain / 应用层的集成

## 1. 为什么需要 vLLM？

LLM 推理的核心瓶颈：**KV Cache 内存管理**

- 每个 token 需要存储 K、V 矩阵（随序列长度线性增长）
- 传统静态分配：预留最大长度内存 → **显存浪费严重**（利用率常 <40%）
- 不同请求序列长度差异大 → 碎片化严重

### PagedAttention（vLLM 核心创新）

受操作系统**虚拟内存/分页**启发：

```
传统方式：
Request A: [KV_0][KV_1][KV_2][  空  ][  空  ][  空  ]  ← 预分配最大长度，浪费
Request B: [KV_0][KV_1][  空  ][  空  ][  空  ][  空  ]

PagedAttention：
Page Pool: [Page_0][Page_1][Page_2][Page_3][Page_4][Page_5]

Request A:  Page_0 → Page_3 → Page_1   (非连续物理块，逻辑连续)
Request B:  Page_2 → Page_4             (按需分配，无浪费)
```

**效果**：显存利用率从 ~40% 提升到 ~90%+，吞吐量提升 **2-24x**。

---

## 2. vLLM 特性一览

| 特性 | 说明 |
|------|------|
| **PagedAttention** | 高效 KV Cache 管理，显存利用率大幅提升 |
| **Continuous Batching** | 动态合并多个请求，GPU 利用率更高 |
| **张量并行** | 多 GPU 横向切分模型（`tensor_parallel_size`） |
| **流水线并行** | 多 GPU 纵向切分层（`pipeline_parallel_size`） |
| **AWQ / GPTQ / FP8 量化** | 减少显存，支持更大模型 |
| **LoRA 热插拔** | 一个 base model + 多个 LoRA 适配器，按请求切换 |
| **OpenAI 兼容 API** | 直接替换 OpenAI API，无缝集成现有应用 |
| **Speculative Decoding** | 草稿模型 + 验证模型，加速 decode 阶段 |
| **Prefix Caching** | 相同 system prompt 的 KV Cache 自动复用 |

---

## 3. 支持的模型

```
LLM：LLaMA 3 / Qwen2.5 / Mistral / DeepSeek / Gemma / Phi
VLM：LLaVA / Qwen2-VL / InternVL / LLaMA-3.2-Vision
MoE：Mixtral / DeepSeek-MoE
```

---

## 4. 部署架构

```
客户端请求（HTTP/gRPC）
        │
        ▼
 vLLM API Server（FastAPI）
        │
        ▼
 LLMEngine（核心调度器）
  ├── 请求队列管理
  ├── Continuous Batching
  └── PagedAttention KV Cache
        │
        ▼
 Worker（GPU）
  ├── 模型前向传播
  ├── Tensor Parallel 通信（NCCL）
  └── Sampler（采样策略）
```

---

## 5. 量化与多 GPU 部署

### 量化选择

| 方案 | 显存节省 | 速度 | 精度 | 推荐度 |
|------|---------|------|------|--------|
| FP16 | 基准 | 基准 | 最高 | ✅ 默认 |
| AWQ int4 | ~4x | +20-30% | 极小损失 | ✅ 推荐量化 |
| GPTQ int4 | ~4x | 略低 | 极小损失 | ✅ 常用 |
| FP8 | ~2x | 最快 | 极小 | ✅ A100/H100 |

### 多 GPU 策略
- **Tensor Parallel**（推荐）：同一模型层横向切分到多 GPU，适合推理延迟敏感场景
- **Pipeline Parallel**：不同层放不同 GPU，适合模型超大、单 GPU 放不下的情况

```python
"""
vLLM 使用示例
安装: pip install vllm  (需要 NVIDIA GPU + CUDA >= 12.1)
文档: https://docs.vllm.ai
"""

# ===== 1. Offline Batch 推理（最简单）=====
try:
    from vllm import LLM, SamplingParams

    # 加载模型（会下载权重，首次较慢）
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,          # GPU 数量
        max_model_len=8192,              # 最大上下文长度
        gpu_memory_utilization=0.9,      # 显存利用率上限
        dtype="bfloat16",                # 精度
        # quantization="awq",            # 启用 AWQ INT4 量化
        # enable_prefix_caching=True,    # 启用 prefix cache
    )

    # 采样参数
    params = SamplingParams(
        temperature=0.7,       # 多样性（0=贪婪，1=随机）
        top_p=0.9,             # nucleus sampling
        max_tokens=512,        # 最大生成 token 数
        stop=["<|im_end|>"],   # 停止词
    )

    # 批量推理（自动 batching，效率最高）
    prompts = [
        "请解释什么是强化学习，用100字以内。",
        "Python 中如何实现单例模式？",
        "解释 PagedAttention 的核心思想。",
    ]

    # 构造对话格式（Qwen2.5 Chat template）
    chat_prompts = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in prompts
    ]

    outputs = llm.generate(chat_prompts, params)

    for i, output in enumerate(outputs):
        print(f"\n[Q{i+1}]: {prompts[i]}")
        print(f"[A{i+1}]: {output.outputs[0].text.strip()}")
        print(f"  生成 {len(output.outputs[0].token_ids)} tokens, "
              f"耗时 {output.metrics.finished_time - output.metrics.first_scheduled_time:.2f}s")

except ImportError:
    print("vLLM 未安装。安装命令:")
    print("  pip install vllm")
    print("  # 或指定 CUDA 版本：")
    print("  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121")

# ===== 2. OpenAI 兼容 API 服务器启动方式 =====
print("\n===== OpenAI 兼容 API 服务器 =====")
server_cmd = """
# 终端中启动 API 服务
python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --port 8000 \\
    --tensor-parallel-size 2 \\            # 2 GPU 张量并行
    --max-model-len 32768 \\
    --gpu-memory-utilization 0.9 \\
    --enable-prefix-caching \\             # 公共 prefix 缓存
    --quantization awq                     # INT4 量化（需 AWQ 模型）
"""
print(server_cmd)

# ===== 3. 调用 OpenAI 兼容 API =====
print("===== 客户端调用（兼容 OpenAI SDK）=====")
client_code = """
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# Chat Completion
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "解释 Transformer 的 Self-Attention 机制。"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,        # 流式输出
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# 查询模型列表
models = client.models.list()
print([m.id for m in models.data])
"""
print(client_code)

# ===== 4. LoRA 热插拔 =====
print("===== LoRA 热插拔 =====")
lora_code = """
from vllm import LLM
from vllm.lora.request import LoRARequest

# 加载 base model（启用 LoRA 支持）
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64,
    max_num_seqs=256,
)

# 不同请求使用不同 LoRA（一个 base model，多个 LoRA）
outputs = llm.generate(
    ["写一首关于春天的诗", "Write a short story about robots"],
    sampling_params=SamplingParams(max_tokens=200),
    lora_request=[
        LoRARequest("chinese_writer", 1, "/lora/chinese_writer"),  # 中文写作 LoRA
        LoRARequest("english_writer", 2, "/lora/english_writer"),  # 英文写作 LoRA
    ]
)
"""
print(lora_code)

# ===== 5. 性能调优建议 =====
print("===== 性能调优 Checklist =====")
tips = """
✅ 启用 FlashAttention（vLLM 默认）
✅ 设置合适的 gpu_memory_utilization（0.85-0.95）
✅ 使用 AWQ/FP8 量化减少显存占用
✅ enable_prefix_caching=True（有共同 system prompt 时收益显著）
✅ max_num_seqs 根据显存调整（控制最大并发请求数）
✅ 生产部署用 guided decoding 限制输出格式（JSON mode）
✅ 多 GPU 用 tensor_parallel_size（同机）；跨机用 pipeline_parallel
"""
print(tips)
```
