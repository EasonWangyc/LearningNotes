"""
PagedAttention 的核心本质就是两件事：

物理显存池（K/V Block Pool）： 将整块 GPU 显存预先切分成固定大小的 Page（例如每个 Block 存放 16 个 Token）。

逻辑到物理的映射（Block Table）： 序列在逻辑上是连续的（Token 0, 1, 2...），但物理上存放在被打散的任意空闲物理块里。在计算 Attention 时，根据 Block Table 把散落在各处的 K/V 动态组装参与运算。
"""
import math
import torch
import torch.nn.functional as F

class KVCacheManager:
    """
    PagedAttention 物理现存与逻辑页表管理器
    """
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, device="cuda") -> None:
        # GPU 上总共划分出的物理 Block 数量
        self.num_blocks = num_blocks
        # 每个block存放的Token数量（如16）
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # 1. 预先在 GPU 上分配固定大小的物理显存池 (Physical Block Pool)
        # 大小：[num_blocks, block_size, num_heads, head_dim]
        self.k_cache_pool = torch.zeros((num_blocks, block_size, num_heads, head_dim), device=device)
        self.v_cache_pool = torch.zeros((num_blocks, block_size, num_heads, head_dim), device=device)

        # 2. 空闲块列表
        self.free_blocks = list(range(num_blocks))

        # 3. 记录每个请求的元数据
        self.block_tables = {}   # req_id -> [物理块ID列表]
        self.req_seq_lens = {}  # req_id -> 当前已生成的总Token长度

    def allocate_request(self, req_id: int):
        """
        为新请求初始化页表
        """
        assert req_id not in self.block_tables, f"请求{req_id}已存在"
        self.block_tables[req_id] = []
        self.req_seq_lens[req_id] = 0

    def append_slot(self, req_id: int):
        """
        核心逻辑：当生成一个新token时，分配物理槽位
        如果当前最后一个block满了， 向空闲池申请新block
        返回：(physical_block_id, block_offset)
        """
        cur_len = self.req_seq_lens[req_id]

        # 检查是否需要新开辟block，条件为整数倍的block_size
        if cur_len % self.block_size == 0:
            assert len(self.free_blocks) > 0, "GPU OOM!"
            new_block_id = self.free_blocks.pop(0)
            self.block_tables[req_id].append(new_block_id)

        # 获取该token在显存中的具体位置
        physical_block_id = self.block_tables[req_id][-1]
        block_offset = cur_len % self.block_size

        # 序列长度+1
        self.req_seq_lens[req_id] += 1
        return physical_block_id, block_offset

    def write_kv(self, req_id: int, k: torch.Tensor, v: torch.Tensor):
        """
        将单步Decode生成的一个Token的K,V写入物理显存池
        K,V形状：(num_heads, head_dim)
        """
        p_block_id, offset = self.append_slot(req_id)
        self.k_cache_pool[p_block_id, offset] = k
        self.v_cache_pool[p_block_id, offset] = v

    def free_request(self, req_id: int):
        """
        请求结束，释放占用的所有物理block
        """
        if req_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[req_id])
            del self.block_tables[req_id]
            del self.req_seq_lens[req_id]

def paged_attention_decode(
    q: torch.Tensor,
    k_cache_pool: torch.Tensor,
    v_cache_pool: torch.Tensor,
    block_table: int,
    seq_len: int,
    block_size: int
):
    """
    单请求decode阶段的PagedAttention算子逻辑
    q:(num_heads, head_dim) -> 当前最新生成的单个token的query
    """
    num_heads, head_dim = q.shape

    # 1. 根据block_table 收集散落在物理池中的各个block
    # 物理块被抽取后，逻辑上拼装为完整的key和value
    # full_k 形状：(num_blocks_for_req, block_size, num_heads, head_dim)
    full_k = k_cache_pool[block_table]
    full_v = v_cache_pool[block_table]
    # 展平
    full_k = full_k.view(-1, num_heads, head_dim)
    full_v = full_v.view(-1, num_heads, head_dim)

    # 2. 截断掉最后一个block里未使用的空白槽位，只保留有效长度seq_len
    valid_k = full_k[:seq_len]
    valid_v = full_v[:seq_len]

    # 3. 维度变换，以进行mha计算
    q = q.unsqueeze(1)                  # (num_heads, 1, head_dim)
    k = valid_k.permute(1, 2, 0)        # (num_heads, head_dim, seq_len)
    v = valid_v.permute(1, 0, 2)        # (num_heads, seq_len, head_dim)

    # 4. 计算注意力score = (Q @ K^T) / sqrt(d_k),(num_heads, 1, seq_len)
    scores = (q @ k) * (1.0 / math.sqrt(head_dim))

    # 5. Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 6. Attn @ V -> (num_heads, 1, head_dim) -> squeeze 得到 (num_heads, head_dim)
    output = (attn_weights @ v).squeeze(1)
    return output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备：{device}")

    # 模拟参数
    num_blocks = 100
    block_size = 4
    num_heads = 4
    head_dim = 64

    # 初始化分页显存管理器
    manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim, device=device)

    # 接收request0，连续自回归生成9个token
    req_id = 0
    manager.allocate_request(req_id)
    print(f"\n--- [请求{req_id}]开始生成 ---")

    for step in range(1, 10):
        q = torch.randn(num_heads, head_dim, device=device)
        k = torch.randn(num_heads, head_dim, device=device)
        v = torch.randn(num_heads, head_dim, device=device)

        manager.write_kv(req_id, k, v)
        out = paged_attention_decode(
            q=q,
            k_cache_pool=manager.k_cache_pool,
            v_cache_pool=manager.v_cache_pool,
            block_table=manager.block_tables[req_id],
            seq_len=manager.req_seq_lens[req_id],
            block_size=block_size
        )

        print(f"Step {step:02d} | 有效长度: {manager.req_seq_lens[req_id]} | "
              f"分配的物理页表: {manager.block_tables[req_id]} | 输出形状: {out.shape}")

    # 模拟请求结束，显存回收
    manager.free_request(req_id)
    print(f"\n--- [请求 {req_id}] 结束，显存回收 ---")
    print(f"剩余空闲物理块数量: {len(manager.free_blocks)}")
    print("✅ PagedAttention 分页调度逻辑模拟成功！")