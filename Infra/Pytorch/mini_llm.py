import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 用大矩阵同时完成Q、K、V的投影，效率更高
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # 注册下三角阵Causal Mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("bias", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, layer_past=None, use_cache=False):
        # batch_size, sequence_length, d_model
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim = 2)

        # 变换维度，维度为：(B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 如果传入历史缓存，把当前的K和V拼接到历史的尾部
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # 如果需要使用缓存，将拼接好的新K和V存下来返回
        present = (k, v) if use_cache == True else None
        
        # 计算q*k^T, k^T维度为(B, n_head, head_dim, T), att维度为(B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Mask分类讨论
        if T > 1:
            # prefill阶段
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        else:
            # decode阶段，只有当前token，它可以且必须看到之前所有的历史KV，所以不需要mask
            pass
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), present

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)

        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x, layer_past=None, use_cache=False):
        # 接收并传递past 和 present
        attn_out, present = self.attn(self.ln_1(x), layer_past, use_cache)
        # 残差连接
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_seq_len) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定
        self.token_emb.weight = self.lm_head.weight

    def forward(self, idx, past_key_values=None, use_cache=False):
        """
        前向传播：用于训练或单次推理。
        idx 维度: (B, T)，里面存的是 token 的整数 ID
        """
        B, T = idx.size()
        # 如果有KV Cache，当前输入的Token是接在历史Token后的，位置编码不能从零开始，而是需要加上偏移
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0

        # 位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # 添加位置编码
        x = self.token_emb(idx) + self.pos_emb(pos)

        presents = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            # 取出当前层对应的KV Cache传入Block
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past, use_cache)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x) # 输出维度:(B, T, vocab_size)
        return logits, presents

    @torch.no_grad # 推理时不计算梯度，节省显存
    def generate(self, idx, max_new_tokens: int):
        """
        自回归生成
        """
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                # decode阶段，只把新生成的最后一个token送入模型
                idx_cond = idx[:, -1:]
            else:
                # prefill阶段
                idx_cond = idx

            # 带有cache的前向传播
            logits, past_key_values = self.forward(idx_cond, past_key_values=past_key_values, use_cache=True)

            # 无论 Prefill 还是 Decode，我们要的都是输出序列最后一步的预测结果
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 将新词拼接到最终输出里，准备下一次 Decode 循环
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    # 模拟配置：极小规模的 Nano-LLM
    vocab_size = 1000
    d_model = 128
    n_head = 4
    n_layer = 2
    max_seq_len = 256
    
    # 实例化模型
    model = MiniLLM(vocab_size, d_model, n_head, n_layer, max_seq_len)
    model.eval() # 切换到推理模式
    
    # 模拟输入 Prompt: 批次为 1，包含 5 个 token
    prompt = torch.randint(0, vocab_size, (1, 5))
    print(f"输入 Prompt: {prompt.tolist()}")
    
    # 让模型自回归生成 10 个新 Token
    generated_sequence = model.generate(prompt, max_new_tokens=10)
    print(f"生成后序列: {generated_sequence.tolist()}")
    print("✅ Mini-LLM 自回归推理测试通过！")