import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将KV的head维度复制n_rep次，使其与Q的head数量对齐
    输入 x:(B, num_kv_heads, T, head_dim)
    输出  :(B, num_q_heads, T, head_dim)
    """
    if n_rep == 1:
        return x

    B, num_kv_heads, T, head_dim = x.shape
    # 利用unsqueeze + expand + reshape实现零额外显存分配的视图变换
    return (
        x[:, :, None, :, :]
        .expand(B, num_kv_heads, n_rep, T, head_dim)
        .reshape(B, num_kv_heads * n_rep, T, head_dim)
    )

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % num_q_heads == 0, "d_model必须能被num_q_heads整除"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads必须能被num_kv_heads整除"

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        # 每组共享 KV 的 Query 头数
        self.num_queries_per_kv = num_q_heads // num_kv_heads  
        self.head_dim = d_model // num_q_heads

        # Q的输出维度是d_model; K和V的输出维度则缩小为num_kv_heads * head_dim
        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # Causal mask 缓存
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor, layer_past=None, use_cache=False):
        B, T, _ = x.shape
        # 1. 获取QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 2. 变换为多头维度
        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # 3. 如果开启kv cache，进行拼接更新（此时缓存的是未经过repeat膨胀的轻量KV）
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        # 4. 在计算Attention矩阵前，将KV 广播到与Q头数相同
        k_expanded = repeat_kv(k, self.num_queries_per_kv)
        v_expanded = repeat_kv(v, self.num_queries_per_kv)

        # 5. 计算Attention
        scores = (q @ k_expanded.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if T > 1:
            total_seq_len = k_expanded.size(-2)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :total_seq_len] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = attn_weights @ v_expanded

        # 6. 拼接输出并通过线性层
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(output), present

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 512
    num_q_heads = 8
    seq_len = 16
    B = 2

    print(f"=== MHA vs GQA vs MQA 对比 (d_model={d_model}, Q_heads={num_q_heads}) ===")

    configs = [
        ("MHA (标准多头)", 8),
        ("GQA (分组查询 - 4组)", 2),
        ("MQA (多查询 - 单头KV)", 1),
    ]

    x = torch.randn(B, seq_len, d_model, device=device)

    for name, num_kv in configs:
        attn = GroupedQueryAttention(d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv).to(device)
        out, cache = attn(x, use_cache=True)
        
        # 统计 KV 投影参数量与 KV Cache 显存大小
        kv_params = sum(p.numel() for p in [attn.k_proj.weight, attn.v_proj.weight])
        cache_k, cache_v = cache
        cache_elements = cache_k.numel() + cache_v.numel()

        print(f"\n【{name}】 (KV Heads = {num_kv}):")
        print(f"  * 输出形状: {out.shape}")
        print(f"  * 单步 KV 投影参数量: {kv_params:,}")
        print(f"  * KV Cache 张量形状: {cache_k.shape}")
        print(f"  * KV Cache 元素总数: {cache_elements} (显存节省率: {(1 - num_kv / num_q_heads) * 100:.1f}%)")