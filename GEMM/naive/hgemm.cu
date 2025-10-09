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
            psum = a[ m * K + k] * b[ k * N + n];
        }
        c[m*N+n] = psum;
    }
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    TORCH_BINDING_COMMON_EXTENSION(hgemm_naive_f16)
    TORCH_BINDING_COMMON_EXTENSION(hgemm_sliced_k_f16_kernel)
}
