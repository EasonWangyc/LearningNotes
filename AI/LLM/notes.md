# 大语言模型学习索引

LLM 相关内容按照模型生命周期拆分为基础、训练、推理和应用四个部分。各主题保持独立笔记，示例图片统一存放在 `resources/` 目录中。

## 内容结构

| 主题 | 内容 |
| --- | --- |
| [基础](./Basics/notes.md) | NLP、Tokenizer、Embedding、位置编码、Normalization、Transformer、Attention、MoE |
| [训练](./Training/notes.md) | LoRA、参数高效微调、量化与精度、数据集、权重装载、Megatron 并行 |
| [推理](./Inference/notes.md) | 解码策略、Prefill、Decoding、KV Cache、Sparse Attention、PagedAttention、线性注意力 |
| [应用](./Applications/notes.md) | RAG、文档切分、BM25、Embedding Retriever、Reranker、Harness Engineering |
| [Agent](./Applications/Agent/notes.md) | Agent Loop、Context、Tools、Skills、Plugins、Hooks、MCP、Subagents、Memory |

## 推荐学习路径

```text
基础
  └── Transformer、Attention、MoE
        ├── 训练：参数、数据、微调、并行
        └── 推理：解码、KV Cache、注意力优化、推理服务
                └── 应用：RAG、Agent、工具调用与工作流
```

## 相关主题

- [AI 基础](../ML/notes.md)：机器学习、深度学习和强化学习
- [推理部署与基础设施](../Infra/notes.md)：CUDA、ONNX、TensorRT、Triton 和 AI Infra
- [Docker](../../Tools/Docker/notes.md)：容器化运行环境与 GPU 服务

## 资源与实验

- `resources/`：Transformer、RoPE、Attention、KV Cache、RAG 等图片和视频资源
- [基础代码实验](./Basics/code.ipynb)：多头注意力和简化模型代码
