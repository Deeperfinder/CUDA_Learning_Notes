import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# """
# 这行代码是整个魔法的起点。让我们分解 load 函数的作用：

# name='minimal_attn':

# 这告诉 PyTorch，你正在创建一个名为 minimal_attn 的 Python 模块。
# 编译完成后，你就可以像 import minimal_attn 一样使用它。
# load 函数的返回值就是这个新创建的模块，所以代码将其赋给了 minimal_attn 变量。
# sources=['main.cpp', 'flash.cu']:

# 这是要编译的源文件列表。
# PyTorch 的扩展工具足够智能，它会识别文件扩展名：
# .cpp 文件会使用 C++ 编译器（如 g++ 或 MSVC）进行编译。
# .cu 文件会使用 NVIDIA 的 CUDA 编译器（nvcc）进行编译。
# extra_cuda_cflags=['-O2']:

# 这是传递给编译器的额外标志（flags）。
# cflags 是 "compiler flags" 的缩写。cuda_cflags 特指传递给 nvcc 的标志。
# -O2 是一个标准的优化级别标志，指示编译器进行积极的性能优化。
# load 函数在幕后做了什么？

# 当这行代码被执行时，PyTorch 会在后台执行一个完整的编译流程：

# 查找编译器: 找到系统上安装的 C++ 编译器和 nvcc。
# 编译: 调用编译器，将 main.cpp 和 flash.cu 编译成目标文件 (.o)。
# 链接: 将编译好的目标文件链接成一个共享库（在 Linux 上是 .so 文件，在 Windows 上是 .pyd 文件）。这个共享库就是一个可以被 Python 动态加载的模块。
# 加载: 将这个新生成的共享库动态加载到当前的 Python 进程中。
# 返回模块: 返回一个代表这个已加载库的 Python 模块对象。
# 这个过程是“即时的”（Just-In-Time），因为它是在你运行 Python 脚本时发生的，而不是预先编译好的。PyTorch 还会缓存编译结果，所以如果你不修改 C++ 代码，下次运行会快得多。
# """
# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu', 'flashv2.cu'], extra_cuda_cflags=['-O2'], verbose=True)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
minimal_attn.forward_V2(q,k,v)
print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
# 这里需要注意的一点是，softmax的维度为-1， 即(b, nh, seq_i, seq_j), 
# 其中i代表着Q的location， J代表K的localtion。那为什么要指定-1呢？
# 因为注意力机制的目的是为每一个查询(Query)生成一个独立的、定制化的权重分布
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
