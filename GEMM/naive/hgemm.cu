#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>  // ✅ 换成 torch/torch.h，它包含了所有需要的头文件
#include <iostream>


__global__ void hgemm_naive_f16_kernel(half* a, half* b, half* c, int M, int N, int K){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N){
        half psum = 0.0;
        #pragma unroll
        for(int k=0; k<K; k++){
            psum += a[ m * K + k] * b[ k * N + n];
        }
        c[m*N+n] = psum;
    }
}


// HGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
template<const int BM=32, const int BN=32, const int BK=32>
__global__ void hgemm_sliced_k_f16_kernel(half *a, half *b, half*c, int M, int N, int K){
    // [1] block Tile: 32x32的block处理c上32x32大小元素的计算
    // [2]     K Tile: 使用共享内存，将K分块为BK大小的块
    __shared__ half s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // tid within block 加载值到shared mem中去，32x32个线程共同协作在行方向上取数据
    // 并且s_a大小为 32x32x4x2=8KB， 使用32x32个线程从global-> shared mem加载32x32 大小的数据
    int tid = threadIdx.y * blockDim.x + tx;

    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;
    
    half sum = __float2half(0.f);
    for (int bk=0; bk < (K + BK - 1) / BK; ++bk){
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        // k x n
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
        #pragma unroll
        for(int k=0; k<BK; ++k){
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();    
    }
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}  

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                \
    m.def(STRINGFY(func), &func, STRINGFY(func));


#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
if(((T).options().dtype() != (th_type))){                      \
    std::cout << "Tensor info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)               \
if ((T).size(0)!=(S0) || ((T).size(1) != (S1))){          \
    throw std::runtime_error("Tensor size mismatch");     \
}

void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int N = a.size(1);
    const int K = b.size(0);
    
    constexpr int BM=32;
    constexpr int BN=32;

    dim3 block(BN, BM);
    dim3 grid((N + BN -1) / BN, (M + BM -1) / BM);

    hgemm_naive_f16_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K
    );
} 

void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM=32;
    constexpr int BN=32;
    constexpr int BK=32;

    dim3 block(BN, BM);
    dim3 grid((N + BN -1) / BN, (M + BM -1) / BM);

    hgemm_sliced_k_f16_kernel<BM, BN, BK><<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K
    );
}

