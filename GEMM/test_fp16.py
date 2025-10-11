import time
import torch
import torch.nn
import torch.utils
import numpy as np

from utils.build_torch_binding import (try_load_gemm_library,
                                       pretty_print_line
                                        )
from typing import Optional

torch.set_grad_enabled(False)

MAX_TFLOPS = -1.0

def print_gemm_result_info(TFLOPS:float,
               out_info:str,
               out_val:list,
               mean_time_secs:float):
    global MAX_TFLOPS
    # calculate TFLOPS improved
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = (TFLOPS - MAX_TFLOPS) / MAX_TFLOPS * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(f"{out_info:>32}: {out_val}, time:{mean_time_secs * 1e3:.2f}ms, TFLOPS:{TFLOPS:<6.2f}(+{improve:.2f}%)")


def run_benchmark(fn:callable, 
                  a: torch.Tensor, 
                  b:torch.Tensor,
                  tag: str, 
                  out: Optional[torch.Tensor]=None,
                  stages: int = -1, 
                  swizzle: bool = False,
                  swizzle_stride: int = 1,
                  num_warmups: int = 1, 
                  num_tests: int = 30,
                  show_all: bool = False):
    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if out is not None:
        out.fill_(0)
    
    if "cublas" in tag:
        gemm.init_cublas_handle()

    # Flush L2 cache with 256 MB data
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        if stages > 1:
            fn(a, b, out, stages, swizzle, swizzle_stride)
        else:
            fn(a, b, out)

    torch.cuda.synchronize()
    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Flush L2
        cache.zero_()
        
        # Record
        if stages > 1:
            start_events[i].record()
            fn(a, b, out, stages, swizzle, swizzle_stride)
            end_events[i].record()

        else:
            start_events[i].record()
            fn(a, b, out)
            end_events[i].record()

    torch.cuda.synchronize()
    total_times_secs = np.array([s.elapsed_time(e)*1e-3 for s, e in zip(start_events, end_events)])[1:]
    mean_time_secs = np.average(total_times_secs)
    out_info = f"{tag}"
    out_flat = out.flatten()
    out_val_first = out_flat[:2].detach().cpu().numpy().tolist()
    out_val_last = out_flat[-2:].detach().cpu().numpy().tolist()
    out_val = [out_val_first[0], out_val_last[-1]]
    out_val = [round(v,8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    # 1 TFLOPS = 10^12 FLOPS
    # ref: https://imgtec.eetrend.com/blog/2021/100062210.html
    TFLOPS = (2 * M * N * K) * 1e-12 / mean_time_secs
    print_gemm_result_info(TFLOPS, out_info, out_val, mean_time_secs)
    torch.cuda.synchronize()
    if "cublas" in tag:
        gemm.destroy_cublas_handle()
    
    del out_flat, cache
    # 释放torch中显存池子中的显存
    torch.cuda.empty_cache()
    
    return out, mean_time_secs

if __name__ == "__main__":
    gemm = try_load_gemm_library(verbose=False)
    Ms = [1024, 2048, 4096, 8192]
    Ns = [1024, 2048, 4096, 8192]
    Ks = [512, 1024, 4096, 8192]

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    # for(M, N, K) in MNKs:
    #     print(f"M={M}, N={N}, K={K}")
    for(M, N, K) in zip(Ms, Ns, Ks):
        MAX_TFLOPS = -1
        pretty_print_line()
        pretty_print_line(f"M={M}, N={N}, K={K}")
        a = torch.randn((M, K)).cuda().half().contiguous()
        b = torch.randn((K, N)).cuda().half().contiguous()
        c = torch.randn((M, N)).cuda().half().contiguous()
        run_benchmark(gemm.hgemm_naive_f16, a, b, "(naive_fp16)", c)
        run_benchmark(gemm.hgemm_sliced_k_f16, a, b, "(sliced_k_fp16)", c)
        run_benchmark(gemm.hgemm_mma_stages_block_swizzle_tn_cute, a, b, "tn(cute+stage4+swizzle<smem>)", c, stages=4)
        run_benchmark(gemm.hgemm_cublas_tensor_op_nn, a, b, "(cublas)", c)
        pretty_print_line()
