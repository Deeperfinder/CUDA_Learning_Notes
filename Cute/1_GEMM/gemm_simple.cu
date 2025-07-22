#include <cute/tensor.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <stdlib.h>

template<typename T>
void gen_rand_data(T *data, int n);

template<typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, T *Aptr, T *Bptr, int m, int n, int k){
    using namespace cute;
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    // Tile 级别分解
    // 坐标为第四象限,这里gB的make_coord(ix, _)即为取出B tile中的一行，其shape 为(kTileN, k)
    // 或者可以记为(kTileN, kTileK, num_tile_k)
    // 注意这里的"_" 符号为切分第ix 行，然后以KtileM和KtileK为小矩阵shape，重复按照这个shape切分多个，获得num_tile_k个tile
    /*
    exmaple:
        tensor A layout:
            ptr[16b](0x56237c284640) o (4,8):(8,1):
                0.00    1.00    2.00    3.00    4.00    5.00    6.00    7.00
                8.00    9.00   10.00   11.00   12.00   13.00   14.00   15.00
                16.00   17.00   18.00   19.00   20.00   21.00   22.00   23.00
                24.00   25.00   26.00   27.00   28.00   29.00   30.00   31.00

        local_tile(A, make_tile(2,2),make_coord(0,0)):
            ptr[16b](0x56237c284640) o (2,2):(8,1):
                0.00    1.00
                8.00    9.00
        
        local_tile(A, make_tile(2,2),make_coord(0,_));
            ptr[16b](0x56237c284640) o (2,2,4):(8,1,2):
                0.00    1.00
                8.00    9.00
            ----------
                2.00    3.00
                10.00   11.00
            ----------
                4.00    5.00
                12.00   13.00
            ----------
                6.00    7.00
                14.00   15.00
    */
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    // thread级别分解
    // MMA: TiledMMA一次能做的矩阵运算所需要的数据
    // MMA_M, MMA_K 表示(kTileM, kTileK)按照TiledMMA能力划分的时候，M方向和K方向需要重复多少次TiledMMA才能完成该矩阵乘法                                                                                  

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(                        );    //(MMA, MMA_M, MMA_K)
    auto tBgB = thr_mma.partition_B(gB);    //(MMA, MMA_N, MMA_K)
    auto tCgC = thr_mma.partition_C(gC);    //(MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));
    
    clear(tCrC);
    // 即有多少个k需要在一个block中进行计算。
    int num_tile_k = size<2>(gA); 
#pragma unroll 1
    for(int itile=0; itile < num_tile_k; ++itile){
        // Global mem -> register mem
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }
    cute::copy(tCrC, tCgC);
}

int main(){
    srand(10086);
    // 初始化数据结构
    using namespace cute;
    using T = half_t;

    T *Cptr;
    T *Aptr;
    T *Bptr;

    int m = 81920;
    int n = 256;
    int k = 256;
    // 传入指针以便修改Aptr的内存地址
    cudaMalloc(&Aptr, sizeof(T) * m * k);
    cudaMalloc(&Bptr, sizeof(T) * n * k);
    cudaMalloc(&Cptr, sizeof(T) * m * n);

    T *Cptr_host;
    T *Aptr_host;
    T *Bptr_host;

    Aptr_host = (T*)malloc(sizeof(T) * m * k);
    Bptr_host = (T*)malloc(sizeof(T) * n * k);
    Cptr_host = (T*)malloc(sizeof(T) * m * n);

    gen_rand_data(Aptr_host, m*k);
    gen_rand_data(Bptr_host, n*k);

    cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

    // 定义了一个Tensor core矩阵乘加操作类型，
    // 16 * 8 * 16（MNK）：-M=16, -K=16, -N=8
    // F16F16F16F16: 表示输入A(F16), 输入B（F16), 累加器C（F16）, 输出D(F16) 均为半精度浮点数 
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    // 定义如何将一个更大的MMA操作分解成由Tensor Core或者其他硬件单元执行的更小的原子MMA操作
    // A 矩阵分块(M/2, K/2)
    // B 矩阵分块(M, K/2)
    using MMA = decltype(make_tiled_mma(mma_atom{},
                        make_layout(Shape<_2, _2, _1>{}),
                        make_layout(Shape<_1, _2, _1>{})));
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;
    
    // (128, 1, 1)
    dim3 block(size(MMA{}));
    dim3 grid(n / kTileN, m / kTileM);
    for(int i=0; i < 5; ++i){
        gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
    }
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("err = %d, str = %s \n", err, cudaGetErrorString(err));

    // cublas
    T *Cptr_cublas;
    cudaMalloc(&Cptr_cublas, sizeof(T) *m *n);
    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = half(1.f);
    half beta = half(0.f);
    for (int i = 0; i < 5; ++i) {
        cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                (half *)Bptr, k,
                (half *)Aptr, k,
                &beta,
                (half *)Cptr_cublas, n);
        if (ret != CUBLAS_STATUS_SUCCESS) {
        printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
        }
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    
    T *Cptr_cublas_host;
    Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

    //compare 
    cudaMemcpy(Cptr_host, Cptr, sizeof(T) *m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T)*m*n, cudaMemcpyDeviceToHost);

    
    float threshold = 0.1;
    for (int i = 0; i < m * n; ++i) {
        float v1 = Cptr_host[i];
        float v2 = Cptr_cublas_host[i];
        if (fabs(v2 - v1) > threshold) {
        printf("v1 = %f, v2 = %f\n", v1, v2);
        }
    }

    Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
    Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

    auto tile = make_tile(8, 8);
    auto coor = make_coord(0, 0);
    Tensor tc1 = local_tile(tensor_C, tile, coor);
    Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

    print_tensor(tc1);
    print_tensor(tc1_cublas);
}
template<typename T>
void gen_rand_data(T *data, int n){
    for(int i=0; i<n; i++){
        float v = (rand() % 200 -100) * 0.01;
        data[i] = v;
    }
}