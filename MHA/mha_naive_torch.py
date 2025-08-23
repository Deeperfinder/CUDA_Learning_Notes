import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout:float=0.0, 
                 bias:bool=True):
        """
            初始化函数
            Args:
            embed_dim: 输入的维度
            num_heads: 多头注意力的头数
        """
        super(MultiHeadAttention,self).__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义一个Q,K,V的线性变换层
        # 使用一个大的线性层，然后分割
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)

        # 输出的线性层
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def _scaled_dot_product_attention(self, q,k,v, mask:None):
        """
            SPDA attn
        """
        # (b, h, s, h_d) * (b, h, hd, s) ==> (b, h, s, s) 
        d_k = q.size(-1)
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attn_score = attn_score.mask_fill(mask,float("-inf"))
        attn = F.softmax(attn_score, dim=-1)
        # (b,h,s,s)*(b,h,s,h_d)
        out = torch.matmul(attn, v)
        return out, attn
    
    def _spilt_heads(self, x, batch_size):
        """
        将输入张量分割成多个头
        Args:
            x: 输入张量
            batch_size: 批次大小
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        # 变换维度，使其变为(batch_size, num_heads, sqe_len, head_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_V(value)
        
        Q = self._spilt_heads(Q, batch_size=batch_size)
        K = self._spilt_heads(K, batch_size=batch_size)
        V = self._spilt_heads(V, batch_size=batch_size)

        # context 是加权后的V， attn_weights 是注意力分数
        context, attn_weights = self._scaled_dot_product_attention(Q,K,V, None)

        # 合并多个头
        # (b,h,s,d) ==> (b,s,h,d)
        context = context.transpose(1,2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)

        # 最终线性层
        output = self.W_O(context)
        
        return output, attn_weights


if __name__ == "__main__":
    num_heads = 40
    head_dim = 128
    embed_dim = 5120
    seq_len = 1000
    batch_size = 20
    
    print("--- 测试自定义 MultiHeadAttention ---")

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)

    output, attn_weights = mha.forward(query, key, value)
    # 打印输出和权重形状
    print(f"输入 Query 的形状: {query.shape}")
    print(f"输出 Output 的形状: {output.shape}")
    # 权重形状: (batch_size, num_heads, seq_len, seq_len)
    print(f"注意力权重的形状: {attn_weights.shape}")
    print("\n" + "="*30 + "\n")


    # --- 计时参数 ---
    warmup_steps = 10
    test_steps = 100

    # --- 设备选择 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # --- 创建模拟输入数据 ---
    query = torch.rand(batch_size, seq_len, embed_dim).to(device)
    key = torch.rand(batch_size, seq_len, embed_dim).to(device)
    value = torch.rand(batch_size, seq_len, embed_dim).to(device)

    # 1. 初始化和准备自定义的 MHA
    custom_mha = MultiHeadAttention(embed_dim, num_heads).to(device)
    custom_mha.eval() # 设置为评估模式

    # 2. 初始化和准备 PyTorch 内置的 MHA
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
    torch_mha.eval() # 设置为评估模式

    # --- 禁用梯度计算 ---
    with torch.no_grad():
        # --- 测试自定义 MHA ---
        print("\n--- 正在测试自定义 MultiHeadAttention ---")
        # 预热
        for _ in range(warmup_steps):
            _ = custom_mha(query, key, value)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式计时
        start_time = time.perf_counter()
        for _ in range(test_steps):
            _ = custom_mha(query, key, value)
        
        # 等待所有CUDA核心完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        custom_mha_time = (end_time - start_time) / test_steps * 1000 # 转换为毫秒
        print(f"自定义 MHA 平均耗时: {custom_mha_time:.4f} ms")


        # --- 测试 PyTorch 内置 MHA ---
        print("\n--- 正在测试 PyTorch 内置 nn.MultiheadAttention ---")
        # 预热
        for _ in range(warmup_steps):
            _ = torch_mha(query, key, value)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 正式计时
        start_time = time.perf_counter()
        for _ in range(test_steps):
            _, _ = torch_mha(query, key, value)

        # 等待所有CUDA核心完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        torch_mha_time = (end_time - start_time) / test_steps * 1000 # 转换为毫秒
        print(f"PyTorch 内置 MHA 平均耗时: {torch_mha_time:.4f} ms")

        # --- 输出对比结果 ---
        print("\n" + "="*30)
        print("性能对比:")
        if torch_mha_time < custom_mha_time:
            speedup = (custom_mha_time - torch_mha_time) / torch_mha_time
            print(f"PyTorch 内置实现比自定义实现快 {speedup:.2f} 倍。")
        else:
            speedup = (torch_mha_time - custom_mha_time) / custom_mha_time
            print(f"自定义实现比 PyTorch 内置实现快 {speedup:.2f} 倍。")
        print("="*30)


