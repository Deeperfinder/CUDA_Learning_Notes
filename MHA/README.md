在A10卡上结果相差不大：
```bash
--- 测试自定义 MultiHeadAttention ---
输入 Query 的形状: torch.Size([20, 1000, 5120])
输出 Output 的形状: torch.Size([20, 1000, 5120])
注意力权重的形状: torch.Size([20, 40, 1000, 1000])

==============================

使用的设备: cuda

--- 正在测试自定义 MultiHeadAttention ---
自定义 MHA 平均耗时: 503.9427 ms

--- 正在测试 PyTorch 内置 nn.MultiheadAttention ---
PyTorch 内置 MHA 平均耗时: 519.7625 ms

==============================
性能对比:
自定义实现比 PyTorch 内置实现快 0.03 倍。
==============================
```