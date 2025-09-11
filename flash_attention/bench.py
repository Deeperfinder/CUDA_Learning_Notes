import math
import os
import argparse
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from flash_attn import flash_attn_func
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

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'  # Ampere

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', 
                    sources=['main.cpp', './cudacore/flash.cu', './cudacore/flashv2.cu', './tensorcore/flash_v1_wmma.cu', './tensorcore/flash_v2_cutlass.cu'], 
                    extra_cuda_cflags=['-O2', '-gencode=arch=compute_86,code=sm_86'], 
                    verbose=True)

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def run_with_profiling(q, k, v):
    print('=== profiling manual attention ===')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        manual_result = manual_attn(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    print('=== profiling minimal flash attention === ')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        minimal_result = minimal_attn.forward(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    print('=== profiling minimal flash attention V2 === ')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        fa2_minimal_res = minimal_attn.forward_V2(q, k, v)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Compare results
    print('flash attn1 values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
    print('flash attn2 values sanity check:', torch.allclose(fa2_minimal_res, manual_result, rtol=0, atol=1e-02))

def bench(fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)

def official_attn(q, k, v):
    """
    调用 FlashAttention 官方实现（FlashAttention-2 推荐函数）
    
    参数:
        q: (batch_size, seqlen, num_heads_q, head_dim)
        k: (batch_size, seqlen_k, num_heads_kv, head_dim)
        v: (batch_size, seqlen_k, num_heads_kv, head_dim)

    返回:
        out: (batch_size, seqlen, num_heads_q, head_dim)
    """
    # 设置参数（你可以根据需要修改）
    dropout_p = 0.0        # 通常 benchmark 关闭 dropout
    causal = False         # 是否启用 causal attention
    window_size = (-1, -1) # 不使用局部窗口
    softcap = None         # 无 soft cap
    alibi_slopes = None    # 不使用 ALiBi
    deterministic = False  # 可设为 True 保证数值可复现

    # 直接调用官方 FlashAttention 函数（支持 GQA/MQA）
    # 自动处理 head 数量不一致（如 GQA）
    out = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        # softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=False  # benchmark 时只关心输出和速度
    )
    return out

def run_without_profiling(q, k, v):
    # used for ncu profiling
    for _ in range(3):
        manual_result = manual_attn(q, k, v)
        minimal_result = minimal_attn.forward(q, k, v)
        torch.cuda.synchronize()

        fa2_minimal_res = minimal_attn.forward_V2(q, k, v)

    # Compare results
    # print('flash attn1 values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
    # print('flash attn2 values sanity check:', torch.allclose(fa2_minimal_res, manual_result, rtol=0, atol=1e-02))
def run_with_timing(q, k, v, q_h, k_h, v_h):
    # # Compare results
    # print('flash attn1 values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
    # print('flash attn2 values sanity check:', torch.allclose(fa2_minimal_res, manual_result, rtol=0, atol=1e-02))

    print("--------------------------------------------------------------------------------------------")
    print("benchmarking flash attention...")
    manual_avg_time = bench(lambda: manual_attn(q, k, v))[0]
    flash_avg_time = bench(lambda: minimal_attn.forward(q, k, v))[0]
    flash_v2_avg_time = bench(lambda: minimal_attn.forward_V2(q, k, v))[0]
    flash_v1_tensorcore_time = bench(lambda: minimal_attn.forward_v1_tensorcore(q, k, v, True))[0]
    flash_v2_cutlass_time = bench(lambda: minimal_attn.forward_v2_cutlass(q, k, v))[0]
    # ✅ 添加官方 FlashAttention
    official_avg_time = bench(lambda: official_attn(q_h, k_h, v_h))[0]

    print("[manual attention]: %.3f ms" % (manual_avg_time * 1000))
    print("[flash attention]: %.3f ms" % (flash_avg_time * 1000))
    print("[flash attention v2]: %.3f ms" % (flash_v2_avg_time * 1000))
    print("[flash attention v1 tensorcore]: %.3f ms" % (flash_v1_tensorcore_time * 1000))
    print("[flash attention v2 cutlass]: %.3f ms" % (flash_v2_cutlass_time * 1000))
    print("[OFFICIAL flash attn]: %.3f ms" % (official_avg_time * 1000))

def main(args):
    # Use small model params, otherwise slower than manual attention. See caveats in README.
    batch_size = 16
    n_head =32
    seq_len = 1024
    head_embd = 64

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    q_h = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float16).cuda()
    k_h = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float16).cuda()
    v_h = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float16).cuda()

    if args.prof:
        run_with_profiling(q, k, v, q_h, k_h, v_h)
    else:
        # run_without_profiling(q, k, v)
        run_with_timing(q, k, v, q_h, k_h, v_h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention benchmark with or without profiling.")
    parser.add_argument("--prof", action="store_true", help="Enable profiling mode.")
    args = parser.parse_args()

    main(args)