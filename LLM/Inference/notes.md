# 大语言模型推理

在经过训练过后的大模型中，所有的权重矩阵、LayerNorm参数、Embedding表等都已经确定下来，模型可以根据输入的文本进行推理(inference)，即根据输入的文本生成相应的输出文本。

<p align="center">
  <img src="../resources/Logits.png" width="100%">
</p>

模型推理到最后输出的是logits（通过softmax得到token的概率分布），要得到token，还需通过decoding strategy（解码策略）将logits转换为token id，再通过tokenizer转换为文本。

## LLM的解码(decoding)

### 不同的解码策略

#### 贪心解码(Greedy Decoding)

每次直接选择概率最高的token，简单高效，但并非全局最优，相当于Top-k中的k=1。

#### 采样(Sampling)

按一定的采样策略选择一个单词，增加生成过程的多样性，但可能会导致生成的文本不连贯。

#### Beam Search

通过维护一个长度为k的候选序列集，每一步(单token推理)从每个候选序列的概率分布中选择概率最高的k个token，再考虑序列概率，保留最高的k个候选序列（避免随推理过程增加所关注序列的数量呈指数级增长）。

#### top-p采样

核心思路：给定token分布$P(x_i\mid x_{1:i-1})$，top-p集合$V^{(p)}\subset V$，使得$\sum_{x\in V^{(p)}}P(x\mid x_{1:i-1})\geq p$，从$V^{(p)}$中采样。和top-k很像，区别在于在何处对分布进行截断（top-k可以理解为固定截断点，top-p是动态截断点）。

### Temperature

logits本质是未归一化的偏好；模型常常过度自信→输出缺少多样性；只靠top-k/top-p截断无法改变分布“尖锐程度”，难以在“稳定 vs 创造力”之间细调。

Temperature提供一个连续控制杆：既可降低幻觉/重复，也可放开想象力（创造vs稳定，类似基于高斯过程的贝叶斯优化中的exploration和exploitation的区别），因此在实践中常把temperature视为“首个要调”的超参数。

配合beam采样和top-p使用，调整temperature值来控制采样的随机性。

数学形式：

$$\tilde{p}_i = \text{softmax}(z_i / T),T>0$$
  * $T<1$：放大logits差异，分布更“尖锐”，输出更确定
  * $T>1$：压平logits，概率更平均，输出更随机
* 实践经验
  * 0.1-0.5：摘要/QA等需要确定性的任务
  * 0.7-1.3：创作/头脑风暴
  * $T\rightarrow 0$: 接近贪婪解码；$T\rightarrow \infty$: 接近均匀采样

### Penalty

* 纯靠temperature/top-k/top-p仍可能出现短循环、口头禅、提示词泄露等模式崩溃
* 不同任务对“重复”和“长度”容忍度不同，需要有针对性的约束手段
* Penalty机制通过修改logits或得分，打破模型对高频token的偏好，提升可控性
* 有的为“软惩罚”(repetition/presence)，有的为“硬约束”(no_repeat_ngram)，可组合使用

#### 常见Penalty机制

* repetition penalty (HF实现):对生成过的token乘以$1/\text{penalty}$或$\text{penalty}$，惩罚重复；>1.0时抑制循环
* presence / frequency penalty (OpenAI)
  * presence：是否出现过→每次出现扣常数
  * frequency：出现次数越多扣得越多→抑制关键词刷屏
* length penalty (Beam Search)
  * 调整对长序列的偏好，$\text{score}/((5+|y|)^\alpha / (5+1)^\alpha)$

## LLM推理的两大阶段

推理的基本实现是NTP模式，即Next Token Prediction。基于LLM自回归生成(autoregressive generation)的特点如下：

- 逐token生成，生成的token依赖于前面的token（生成i个token后，将1~i个token作为上下文继续生成第i+1个token）；
- 一次只能生成一个token，无法同时生成多个token。

Deepseek v3引入了MTP的机制，主要是应用在训练过程中。

### 第一阶段：Prefill Phase

当用户向大模型发送一段 Prompt时，模型首先需要“阅读”并理解这段完整的输入。这一步称为 Prefill Phase（预填充阶段）。在这个阶段，模型会将整个 Prompt 作为输入，一次性处理完毕，生成对应的隐藏状态（hidden states）和注意力缓存（Key/Value caches）。

在NTP过程中，模型需要不断地处理和存储历史上下文信息（Key/Value缓存），以便在生成下一个token时参考之前的内容。随着生成的token数量增加，Key/Value缓存的大小也会线性增长，导致显存占用和计算开销显著增加。而通过Prefill Phase，模型可以一次性处理完整的Prompt，提前计算并存储必要的上下文信息，从而在后续的token生成过程中减少重复计算，提高推理效率。

### 第二阶段：Decoding Phase

当第一个"下一个token"生成完毕后，LLM开始"自回归推理"生成。

第二个"下一个token"：输入x的shape: $(b,s+1,h)$，计算开销$O((s+1)^2)$

第三个"下一个token"：输入x的shape: $(b,s+2,h)$，计算开销$O((s+2)^2)$

第n个"下一个token"：输入x的shape: $(b,s+n-1,h)$，计算开销$O((s+n-1)^2)$

### 两阶段分析

Prefill阶段 与 Decode阶段 具有截然不同的计算与访存特性，它们对硬件存储介质（主要是GPU显存）的读写交互方式也完全不同。Prefill阶段并行处理整个输入提示词，属于计算密集型（Compute-Bound）任务，对存储主要执行“一次性批量写入KV Cache”；Decode阶段自回归逐个生成Token，属于访存密集型（Memory-Bound）任务，对存储持续执行“频繁读取历史KV Cache并追加写入少量新KV”的操作。

#### Prefill 阶段的特点与存储交互

特点：
- 并行处理：用户输入的整段Prompt（所有Token）在这一阶段同时输入模型。
- 计算密集：算力（Tensor Core）能被高度打满，计算复杂度随输入长度呈平方级（\(O(N^2)\)）上升。
- 核心指标：决定了 TTFT（Time to First Token，首字延迟）。

与存储介质的交互方式是大批量连续写入（Write-heavy / High Throughput Write），模型在计算每一层的注意力（Attention）时，把输入序列中所有Token对应的 Key (K) 和 Value (V) 矩阵 计算出来，一次性高并发地写入 到显存的 KV Cache 区域中。这种交互对显存带宽的压力相对平缓，更多依赖核心计算能力。

#### Decode 阶段的特点与存储交互

特点：
- 自回归逐个生成：模型基于已有上下文，一步步环形迭代，每次只吐出一个新Token。
- 访存密集：由于每次计算只处理1个新Token，运算量极小，算力单元大部分时间在“等数据”，瓶颈卡在 显存带宽（Memory Bandwidth） 上。
- 核心指标：决定了 TPOT（Time Per Output Token，每字生成速度/流畅度）。

与存储介质的交互方式是高频次、零散的读取与追加写入（Read-heavy & Append Write）：
- 读：每生成一个新Token，计算单元需要把显存中所有历史Token的KV Cache全部重新读入处理器中，与当前新Token的Query进行矩阵乘法。
- 写：当前新Token计算完成后，自身产生的最新KV数据会被追加写入显存的KV Cache末尾。

这种模式导致显存带宽被大量占用（每次都要搬运全部历史缓存），如果显存带宽不够，生成速度就会明显变慢。

## KV Cache

考虑一次LLM推理过程中的计算开销，先进行一下符号的规定：
  * b: batch size
  * s: sequence length
  * h: hidden size/dimension
  * nh: number of heads
  * hd: head dimension

给定矩阵$A\in R^{m\times n}$和矩阵$B\in R^{n\times p}$，计算$AB$中的一个元素需要$n$次乘法操作和$n$次加法操作，一共有$mp$个元素，总计算开销为$2mnp$ 。

**Self-attn模块：**

第一步计算: $Q=xW_q$, $K=xW_k$, $V=xW_v$
  * 输入x的shape: $(b,s,h)$，weight的shape: $(h,h)$
  * Shape视角下的计算过程: $(b,s,h)(h,h)\rightarrow(b,s,h)$
  * 如果在此进行多头拆分(reshape/view/einops)，shape变为$(b,s,nh,hd)$，其中$h=bh*hd$
  * 计算开销: $3\times 2bsh^2\rightarrow 6bsh^2$

第二步计算: $O=\text{softmax}(\frac{QK^T}{\sqrt{h}})V$
  * $QK^T$计算: $(b,nh,s,hd)(b,nh,hd,s)\rightarrow (b,nh,s,s)$
  * 计算开销: $2b*nh*s^2*hd=2bs^2h$ (为理解方便，暂且忽略softmax的计算开销)
  * $\text{softmax}(\frac{QK^T}{\sqrt{h}})V$计算: $(b,nh,s,s)(b,bh,s,hd)\rightarrow(b,nh,s,hd)$
  * 计算开销: $2bs^2h$
  * 总计算开销: $4bs^2h$

第三步计算：$x_{\text{out}}= O W_o + x$
  * $O$的shape为$(b,s,h)$，$W_o$的shape为$(h,h)$，计算过程为$(b,s,h)(h,h)\rightarrow(b,s,h)$
  * 计算开销: $2bsh^2$

Self-attn模块总计算开销: $8bsh^2+4bs^2h$

**MLP模块：**

$x=f_\text{activation}(x_{\text{out}}W_{\text{up}})W_{\text{down}}+x_{\text{out}}$
第一步计算，假设上采样到4倍
  * Shape变化:$(b,s,h)(h,4h)\rightarrow(b,s,4h)$
  * 计算开销: $8bsh^2$

第二步计算，假设下采样回1倍
  * Shape变化:$(b,s,4h)(4h,h)\rightarrow(b,s,h)$
  * 计算开销: $8bsh^2$

MLP模块总计算开销: $16bsh^2$

Decoder layer一次推理的总开销：$24bsh^2+4bs^2h$，为$s$的平方级别（$b\ll s$，且h在模型确定后为固定值）。

考虑s和s+1的两种情况下的$QK^T$

每次

视频的直观展示：

without KV Cache:
<video src="../resources/Without KV Cache.mp4" controls="controls" width="100%" height="auto">
</video>

with KV Cache:

<video src="../resources/KV Cache.mp4" controls="controls" width="100%" height="auto">

</video>

所以，真正自回归计算的部分是$(b,s+1,h)$中的第二个维度$index_{s+1}$的部分，复用的是用于计算$(b,s+1,h)$中第二维度$index_{s+1}$的数值，从shape的视角: $(b,s+1,h)\rightarrow (b,1,h)$

### 为什么只有K和V需要缓存

整个self-attn计算过程中，只有$QK^T$中的$K$和$\text{softmax}(\frac{QK^T}{\sqrt(h)})V$中的$V$需要复用，而Q依赖当前token的Embedding，必须实时计算；Attn输出和MLP输出也会被LayerNorm/残差更新，无法直接重用。
### KV Cache的内存消耗

对于批大小 $b$，层数 $l$，头数 $h$，序列长度 $s$，头维度 $d$：
  $$
  \text{KV\_memory} \approx b \times l \times h \times s \times d \times 2 \times \text{dtype\_size} $$

dtype 通常为 FP16/BF16；缓存越大，显存消耗越高，存储和计算均为$O(s)$级别的开销。

## Attention优化

### Sparse Attention

虽然 KV Cache 避免了重复计算，但随着序列变长，Cache 的内存占用和 Attention 的计算量仍呈线性（甚至平方，取决于具体实现）增长。**Sparse Attention** 的核心思想是：**并非所有的 token 都需要关注之前所有的 token**。很多时候，局部上下文或特定的关键信息就足够了。

通过只让 Token 关注一部分历史信息（即让 Attention Matrix 变得稀疏），可以将计算复杂度从 $O(N^2)$ 降低到 $O(N \log N)$ 甚至 $O(N)$。

#### Attention的稀疏性

在 Self-Attention 计算过程中发现，注意力矩阵中，大部分权重接近0，且整体表现出如下几种现象。

<p align="center">
  <img src="../resources/sparse attention.png" width="100%">
</p>

#### Sparse Attention的几种实现方式

##### static pattern

1. Sliding windows: 维护一个固定大小(k)的窗口，保留最近的 tokens 参与计算，其余全部丢弃。

<p align="center">
  <img src="../resources/sliding windows.png" width="100%">
</p>

优点是实现简单，计算复杂度降低到 $O(k)$；缺点是精度损失较大，尤其是在长度超过预训练长度后大幅下降。

2. Attention sinks(StreamingLLM):

[StreamingLLM](https://arxiv.org/abs/2309.17453) 发现注意力权重往往会集中在首 token 上，将这一现象称为 attention sinks。基于该发现，StreamingLLM 在 sliding window 的基础上进一步保留 attention sinks，降低了长文本场景下稀疏导致的精度损失。

<p align="center">
  <img src="../resources/attention sinks.png" width="100%">
</p>

##### dynamic pattern

1. [MInference](https://arxiv.org/abs/2407.02490) 通过观察注意力矩阵，总结出三种常见模式，根据输入动态选择最合适的模式，从而加速 prefill 阶段。

2. [Quest](https://arxiv.org/abs/2406.10774) 采用分页设计，估计每个 KV page 与当前 Q 的相似度，动态选择最相似（激活值最高）的 pages 参与计算。

对比static pattern：

优点是相较于 static pattern，dynamic pattern 类的方法精度更高；

缺点是由于计算最合适的 tokens 会引入一定 overhead，综合下来会比简单的 static pattern 方法慢（但是相比 dense attention 还是有加速效果）;同时，如何设计选择算法也依赖经验（启发式）。

### Paged Attention

**PagedAttention** 是高吞吐量推理框架 [vLLM](https://github.com/vllm-project/vllm) 的核心技术。针对标准 KV Cache 显存利用率低的问题，它借鉴了操作系统中**虚拟内存（Virtual Memory）**和**分页（Paging）**的管理思想。

#### 标准 KV Cache 的显存浪费

在传统的实现中（如 HuggingFace Transformers），KV Cache 通常存储在连续的显存空间中。由于 LLM 生成的长度通常是未知的，为了防止溢出，系统往往需要预分配最大可能长度（max_seq_len）的连续内存。这就导致了两种严重的显存浪费：

*   **内部碎片（Internal Fragmentation）**：预分配了很长的空间，但实际生成的序列很短，多余的空间无法被利用。
*   **外部碎片（External Fragmentation）**：显存中分散着许多小的空闲块，但由于不连续，无法分配给需要大块连续内存的新请求。

据统计，在传统系统中，显存的浪费率可能高达 **60% - 80%**。

#### 核心思想

PagedAttention 允许 KV Cache 在显存中**非连续**存储。它将每个序列的 KV Cache 切分成固定大小的块（**KV Block**），类似于 OS 中的 Page。

*   **Logical Blocks（逻辑块）**：从请求的角度看，token 是连续的。
*   **Physical Blocks（物理块）**：从显存的角度看，数据存储在非连续的物理地址中。
*   **Block Table（页表）**：维护逻辑块到物理块的映射关系。

#### 实现细节

假设 Block Size = 4（每个块存 4 个 token 的 KV）：

1.  **分配（Allocation）**：
    *   当一个新的 token 生成时，如果当前最后的物理块未满，直接写入。
    *   如果已满，调度器从全局的物理块池中申请一个新的物理块（地址可以是任意位置），并在 Block Table 中记录映射。

2.  **注意力计算（Attention Calculation）**：
    *   PagedAttention 编写了定制的 CUDA Kernel。
    *   在计算 Attention Score 时，Kernel 不再假设 KV 是连续的，而是根据 Block Table 动态地去显存的不同位置抓取数据块进行计算。

3.  **内存共享（Memory Sharing）**：
    *   这是 PagedAttention 最大的优势之一。类似于 OS 的写时复制（Copy-on-Write），多个请求可以共享相同的物理块。
    *   **应用场景**：
        *   **Parallel Sampling**：同一个 Prompt 生成多种不同的输出。Prompt 部分的 KV Cache 只需要存储一份。
        *   **Beam Search**：多个 Beam 共享公共的前缀历史。

#### 代码逻辑示意

虽然底层的 PagedAttention 是用 CUDA 实现的，但其 Python 端的调度逻辑大致如下：

```python
class BlockTable:
    def __init__(self):
        self.logical_to_physical = [] # 存储物理块的ID

def allocate_block(physical_block_pool):
    # 从空闲池中取出一个物理块ID
    return physical_block_pool.pop()

# 随着推理进行，动态分配显存
current_token_index = 10
block_size = 4

if current_token_index % block_size == 0:
    # 需要分配新块
    new_physical_id = allocate_block(pool)
    sequence.block_table.append(new_physical_id)

# 将KV写入对应的物理地址
physical_address = map_to_address(new_physical_id)
write_kv(physical_address, k, v)
```
### Linear Attention

标准 Self-Attention 的核心瓶颈在于其 $O(N^2)$ 的时间和空间复杂度。**Linear Attention (线性注意力)** 旨在通过改变计算顺序或近似 Kernel 函数，将复杂度降低到 $O(N)$。

回顾标准 Attention 公式：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

这里必须先计算 $QK^T$，得到一个 $N \times N$ 的矩阵（Attention Map），然后再乘以 $V$。正是这个 $N \times N$ 矩阵导致了平方级复杂度。

#### 核心思想

核心思想是结合律与 Kernel Trick。如果能去掉 Softmax，或者将 Softmax 进行某种分解，我们就可以利用矩阵乘法的**结合律 (Associativity)**。

假设我们可以找到一个映射 $\phi(\cdot)$ 使得 $\text{sim}(Q, K) \approx \phi(Q) \phi(K)^T$，那么：

$$ \text{Attention}(Q, K, V) \approx \left( \phi(Q) \phi(K)^T \right) V $$

利用结合律，我们改变计算顺序：

$$ \phi(Q) \left( \phi(K)^T V \right) $$

*   **传统做法**：$(Q K^T) V$
    *   $Q K^T$: $(N \times d) \times (d \times N) \rightarrow (N \times N)$
    *   Result $\times V$: $(N \times N) \times (N \times d) \rightarrow (N \times d)$
    *   **复杂度**: $O(N^2)$

*   **Linear Attention**：$Q (K^T V)$
    *   $K^T V$: $(d \times N) \times (N \times d) \rightarrow (d \times d)$
    *   $Q \times$ Result: $(N \times d) \times (d \times d) \rightarrow (N \times d)$
    *   **复杂度**: $O(N d^2)$

由于通常 sequence length $N$ 远大于 hidden dimension $d$，因此 $O(N)$ 远优于 $O(N^2)$。

#### Efficient Attention / Performer

难点在于如何处理 Softmax 这种非线性操作。

**Efficient Attention**: 去掉 Softmax，改为分别对 $Q$ 和 $K^T$ 做 Row-wise / Col-wise 的归一化，使得它们可以直接相乘。
    $$ \text{Attention}(Q,K,V) = \rho_q(Q) (\rho_k(K)^T V) $$
**Performer**: 使用随机正交特征 (Random Orthogonal Features) 来近似 Softmax 核函数。

优缺点对比：

*   **优点**：
    *   推理速度快，显存占用极低，特别是对于超长文本。
    *   实现了真正的 $O(N)$ 复杂度。
*   **缺点**：
    *   **精度损失**：由于是对 Softmax 的近似或替换，往往无法完全达到标准 Attention 的表现。
    *   **训练不稳定**：部分 Kernel Trick 方法在训练时不如标准 Attention 稳定。

因此，目前主流 LLM (如 Llama, GPT) 依然坚持使用标准 Attention (配合 FlashAttention 优化)，而 Linear Attention 更多用于特定的长序列任务或作为架构创新的组件 (如 RWKV, Mamba 等其实在思想上与 Linear Attention 有异曲同工之妙——即 RNN 形式的推理)。

### Gated Attention

标准 Transformer 结构通常由 **Multi-Head Attention (MHA)** 和 **Feed-Forward Network (FFN)** 两个独立的子层叠堆而成。**Gated Attention** 的核心思路是引入门控机制（Gated Mechanism，类似于 LSTM 中的门或 GLU），用来更精细地控制信息的流动，或者将 Attention 与 FFN 的功能进行融合。

#### Gated Attention Unit (GAU)
GAU 是在论文 [Transformer Quality in Linear Time](https://arxiv.org/abs/2202.10447) (FLASH) 中提出的结构。
*   **动机**：MHA 极其消耗显存，而且多头之间存在冗余；FFN 参数量大。GAU 试图将两者合二为一，用更少的参数和计算量达到相当的效果。
*   **结构**：
    GAU 并不是简单地堆叠，而是采用了一种“三明治”式的门控结构：
    $$ O = (U \odot \text{Attention}(Z)) W_o $$
    其中：
    *   $U = \phi_u(X W_u)$ 是门控分支（类似 FFN 中的激活）。
    *   $Z$ 及其变换用于计算简化的注意力（通常只需要 1 个 Head，而非 MHA 的多个 Head）。
    *   $\odot$ 是逐元素乘法（Hadamard Product）。

这种结构证明了**单头注意力（Single-head Attention）配合强有力的门控（Gating）**，可以匹敌标准的多头注意力。

#### Gated Linear Attention (GLA)
在使用由 **Linear Attention** 衍生出的现代架构（如 RWKV, RetNet, Mamba）中，"Gated" 的含义往往指引入**时间衰减（Time-decay）**或**数据依赖的门（Data-dependent Gate）**。

在标准 Linear Attention $Q(K^TV)$ 中，历史信息是等权累加的。引入 Gate 后：
$$ h_t = \alpha_t \odot h_{t-1} + K_t^T V_t $$
$$ y_t = Q_t h_t $$
这里 $\alpha_t$ 就是一个遗忘门（Forgot Gate）。这使得模型能够：
1.  **遗忘**：丢弃不重要的历史噪音。
2.  **位置编码**：通过指数衰减 implicitly 包含相对位置信息。

**总结**：Gated Attention 通过乘法门控操作，赋予了模型更强的非线性表达能力，使其能用更简化的注意力形式（如线性注意力或单头注意力）达到标准 Transformer 的性能，通常是实现“线性复杂度”大模型的关键组件。

## 数的精度与模型量化

在大模型推理与部署中，权重和激活值的数值精度直接影响显存占用、推理速度和模型效果。精度越低，显存越省、速度越快，但量化误差也可能越
大。以下是常见的数值精度格式。

### 数的精度表示基础

任何一个浮点数可以表示为：

$$V = (-1)^{\text{sign}} \times \text{mantissa} \times \text{base}^{\text{exponent}}$$

其中 sign 表示符号位，mantissa 表示尾数（决定精度/precision），exponent 表示指数（决定表示范围/range）。不同精度格式在"范围 vs 精
度"之间做不同的取舍。

<p align="center">
  <img src="../resources/numerical_precision_formats.png" width="100%">
</p>

> 上图展示了不同精度格式的 bit 分配方式。来源：[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-vis
ual-guide-to-quantization)。

---

### FP（IEEE 754 浮点数）

IEEE 754 是最通用的浮点数标准，广泛应用于科学计算和深度学习训练。

#### FP32（单精度浮点 / Full Precision）

| 属性 | 值 |
|------|-----|
| 总位数 | 32 bit |
| 符号位 | 1 bit |
| 指数位 | 8 bit |
| 尾数位 | 23 bit |
| 表示范围 | $\approx \pm 3.4 \times 10^{38}$ |
| 最小正数 | $\approx 1.18 \times 10^{-38}$ |
| 精度 | ~7 位十进制有效数字 |

- 模型训练和推理的"黄金标准"，精度最高。
- 缺点：显存占用大，推理速度慢。一个 7B 模型约需 $7\text{B} \times 4\text{ bytes} = 28\text{ GB}$ 显存。

#### FP16（半精度浮点 / Half Precision）

| 属性 | 值 |
|------|-----|
| 总位数 | 16 bit |
| 符号位 | 1 bit |
| 指数位 | 5 bit |
| 尾数位 | 10 bit |
| 表示范围 | $\pm 65504$ |
| 最小正数 | $\approx 6.10 \times 10^{-5}$ |

- 显存需求减半（7B 模型约 14 GB），支持 Tensor Core 加速。
- 局限：表示范围窄，容易出现上溢（overflow）或下溢（underflow）；训练时通常需要 loss scaling 来稳定梯度。
- 推理场景下常与 FP32 混合使用（Mixed Precision）：矩阵乘法用 FP16，累加/归一化用 FP32。

#### FP8（8 位浮点）

FP8 是较新的格式，由 NVIDIA H100（Hopper 架构）首次在硬件层面原生支持。FP8 分两种变体，针对不同用途：

**E4M3（FP8 推理/前向）**

| 属性 | 值 |
|------|-----|
| 总位数 | 8 bit |
| 指数位 | 4 bit |
| 尾数位 | 3 bit |
| 表示范围 | $\pm 448$ |
| 精度 | 更高，适合前向传播和推理 |

**E5M2（FP8 梯度）**

| 属性 | 值 |
|------|-----|
| 总位数 | 8 bit |
| 指数位 | 5 bit |
| 尾数位 | 2 bit |
| 表示范围 | $\pm 57344$ |
| 精度 | 较低但范围大，适合存储梯度（梯度常有极端值） |

#### FP4（4 位浮点）

| 属性 | 值 |
|------|-----|
| 总位数 | 4 bit |
| 指数位 | 2 bit |
| 尾数位 | 1 bit |

- NVIDIA Blackwell 架构（B200/GB200）开始支持 FP4 硬件加速。
- 极致的显存压缩，但量化误差显著增大，通常需要配合特殊的量化策略和校准。

#### TF32（TensorFloat-32）

| 属性 | 值 |
|------|-----|
| 总位数 | 19 bit（实际占用 32 bit 存储） |
| 符号位 | 1 bit |
| 指数位 | 8 bit（与 FP32 相同） |
| 尾数位 | 10 bit（与 FP16 相同） |

- NVIDIA Ampere 架构（A100）引入，在 Tensor Core 中自动使用。
- 本质是 FP32 的范围 + FP16 的精度，输入输出仍为 FP32，但矩阵乘法内部截断为 TF32。
- 训练场景下几乎无精度损失，但速度比 FP32 快 8-10 倍（与 FP16 接近）。

---

### BF（Brain Floating Point）

#### BF16（Google Brain Float）

| 属性 | 值 |
|------|-----|
| 总位数 | 16 bit |
| 符号位 | 1 bit |
| 指数位 | 8 bit |
| 尾数位 | 7 bit |
| 表示范围 | 与 FP32 相同（$\approx \pm 3.4 \times 10^{38}$） |
| 精度 | 低于 FP16（~2 位十进制有效数字 vs ~3 位） |

- Google 为 TPU 设计，现已被广泛支持（NVIDIA A100/H100、AMD MI300 等）。
- **核心优势**：指数位与 FP32 相同（8 bit），表示范围大，训练时不需 loss scaling，溢出风险极低。
- 代价是尾数精度低于 FP16，但大模型训练往往对"范围"的敏感度高于"精度"。
- 目前主流 LLM（Llama、GPT、Gemma 等）的推理和训练均以 BF16 为默认精度。

---

### INT（整数量化）

整数量化将浮点权重和激活值映射到低位整数，利用整数运算的高效性加速推理。

#### INT8

| 属性 | 值 |
|------|-----|
| 总位数 | 8 bit |
| 表示范围（有符号）| $[-128, 127]$ |
| 表示范围（无符号）| $[0, 255]$ |

- 最成熟的整数量化方案，NVIDIA T4/A100/H100 均有原生 INT8 Tensor Core 加速。
- 分为对称量化与分组量化两种基本策略：

**量化基本公式**：

$$X_{\text{int}} = \text{round}\left(\frac{X_{\text{fp}}}{S}\right) + Z$$

其中 $S$ 为 scale（缩放因子），$Z$ 为 zero-point（零点偏移）。

**对称量化 vs 分组量化**：

- **对称量化**：$Z = 0$，即 $X_{\text{int}} = \text{round}(X_{\text{fp}} / S)$。实现简单，要求数据分布以 0 为中心。适合权重（权重
通常呈对称分布）。
- **分组量化**：$Z \neq 0$，可以更好地匹配非对称分布（如激活值经过 ReLU 后全为正），但需要额外存储 zero-point。

**量化粒度**：

| 粒度 | 描述 | Scale 数量 | 精度 |
|------|------|-----------|------|
| Per-Tensor | 整个张量共享一个 scale | 1 | 最低 |
| Per-Token/Per-Channel | 每一行/列一个 scale | $N$ | 适中 |
| Per-Group | 每 $g$ 个元素一组一个 scale | $N/g$ | 最高 |

粒度越细，精度越高，但额外存储的 scale/zero-point 开销也越大。

#### INT4

| 属性 | 值 |
|------|-----|
| 总位数 | 4 bit |
| 表示范围（无符号）| $[0, 15]$ |

- 显存需求降至 FP16 的 1/4，一个 7B 模型仅需约 3.5 GB。
- NVIDIA Blackwell 架构开始支持 INT4 Tensor Core。

#### INT2 / binary

- 极端量化方案，2 bit 或 1 bit 表示一个权重。
- 目前仅用于研究场景，实际大规模部署中的精度损失仍然难以接受。

---

### NF4（NormalFloat 4-bit）

NF4 是 QLoRA（[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)）中提出的专有 4 位量化格式。

**设计动机**：标准 INT4 假设数据均匀分布在取值范围内，但神经网络的权重通常近似服从均值为 0 的正态分布。NF4 将 4 bit 的 16 个量化等
级按正态分布的分位数来分配，使得大部分量化区间集中在高概率密度区域，从而在 4 bit 下最大化信息保留。

| 属性 | 值 |
|------|-----|
| 量化等级数 | 16（4 bit） |
| 分布假设 | 标准正态分布 $\mathcal{N}(0, 1)$ |
| 量化值 | 正态分布的分位数点 |

NF4 量化等级的具体取值（归一化后）：

$$[-1.0, -0.696, -0.525, -0.395, -0.284, -0.188, -0.101, -0.019, \ 0.019, \ 0.101, \ 0.188, \ 0.284, \ 0.395, \ 0.525, \ 0.696, \
1.0]$$

- 高密度区域（0 附近）量化间隔小，低密度区域（尾部）量化间隔大。
- QLoRA 中与双重量化（Double Quantization）结合使用，使得 65B 模型可在单张 48GB GPU 上微调。
- 配合分页优化器（Paged Optimizer）进一步节省显存。

---

### 各精度格式对比总结

| 格式 | 位数 | 指数 | 尾数 | 表示范围 | 典型应用 |
|------|------|------|------|---------|---------|
| FP32 | 32 | 8 | 23 | $\pm 3.4\times 10^{38}$ | 训练金标准 |
| TF32 | 19(存32) | 8 | 10 | $\pm 3.4\times 10^{38}$ | A100+ 训练加速 |
| BF16 | 16 | 8 | 7 | $\pm 3.4\times 10^{38}$ | 主流训练/推理 |
| FP16 | 16 | 5 | 10 | $\pm 65504$ | 推理/混合精度训练 |
| INT8 | 8 | — | — | $[-128,127]$ | 推理量化 |
| FP8 (E4M3) | 8 | 4 | 3 | $\pm 448$ | H100+ 推理 |
| FP8 (E5M2) | 8 | 5 | 2 | $\pm 57344$ | H100+ 训练梯度 |
| INT4 | 4 | — | — | $[0,15]$ | 边缘推理 |
| NF4 | 4 | — | — | 正态分位 | QLoRA 微调 |
| FP4 | 4 | 2 | 1 | $\pm 6$ | Blackwell 推理 |

---

### 训练与推理中的精度选择策略

#### 训练阶段

1. **FP32 全精度训练**：最稳定，但显存和算力需求最高，已较少用于大模型。
2. **FP16 混合精度训练**：前向/反向用 FP16，权重更新维护一份 FP32 主副本。需要 loss scaling 防止梯度下溢。NVIDIA APEX / PyTorch AM
P 均支持。
3. **BF16 混合精度训练**：无需 loss scaling，范围与 FP32 一致，已成为主流选择。NVIDIA Ampere 及以后架构原生支持。
4. **FP8 训练**：NVIDIA H100 引入 Transformer Engine，自动将部分层的矩阵乘法切换为 FP8，配合延迟缩放（delayed scaling）策略，进一
步提高训练吞吐。

#### 推理阶段

1. **FP16/BF16 推理**：当前主流，精度稳定，硬件支持广泛。
2. **INT8 量化推理**：基本无损，吞吐提升约 2 倍，适用于大多数生产场景。
3. **INT4/NF4 量化推理**：显存大幅下降，但需要特定的量化方案（如 GPTQ、AWQ、bitsandbytes）来保持精度。
4. **混合精度推理**：不同层使用不同精度（如注意力层用 FP16，FFN 层用 INT8），在保持精度的前提下最大化吞吐。

---

### 常见量化方法

| 方法 | 原理 | 特点 |
|------|------|------|
| **GPTQ** | 基于近似二阶信息（Hessian），逐列量化 + 补偿剩余误差 | 4 bit 下精度保持好，校准数据量需求大 |
| **AWQ** | 根据激活值大小识别"显著权重"（salient weights），对其保护后再做量化 | 比 GPTQ 更快，保护约 1% 的重要通道 |
| **bitsandbytes** | 在线量化，支持 INT8/NF4 即插即用 | QLoRA 的基础实现，适合快速实验 |
| **GGUF/GGML** | llama.cpp 使用的量化格式 | 支持 CPU 推理，q4_K_M、q5_K_M 等变体丰富 |
| **SmoothQuant** | 将激活值的量化难度通过数学变换"平滑"到权重上 | W8A8 量化，适应激活值异常值 |
| **FP8 量化** | NVIDIA Transformer Engine 原生 | H100+ 硬件支持，训练推理均可 |

---

### 显存占用速算公式

对于参数量为 $P$ 的模型，仅加载参数（不含 KV Cache 和中间激活）所需理论显存为：

| 精度 | 每参数字节数 | 7B 模型显存 | 70B 模型显存 |
|------|-------------|------------|-------------|
| FP32 | 4 bytes | ~26 GB | ~261 GB |
| FP16/BF16 | 2 bytes | ~13 GB | ~130 GB |
| INT8 | 1 byte | ~6.5 GB | ~65 GB |
| INT4/NF4/FP4 | 0.5 byte | ~3.3 GB | ~33 GB |

> 实际部署中还需额外计算显存用于 KV Cache、中间激活和框架开销，通常在理论值的 1.2-1.5 倍。
