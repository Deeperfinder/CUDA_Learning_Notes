#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_V2_kernel(const float* Q, const float* K, 
                                  const float* V, const int N, const int d,
                                  const int Tc, const int Tr, const int Bc, 
                                  const int Br, const float softmax_scale, 
                                  float* l, float* m, float* O){
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;
    // 1.找到是哪个batch的哪个head
    int qkv_offset = bx * gridDim.y *  N * d + by * N * d;
    int lm_offset = bx * gridDim.y * N + by * N;

    // 2. 定义sram， 运行时候动态指定
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float *Qi = sram;
    float *Oi = &sram[tile_size];
    float *Kj = &sram[tile_size*2];
    float *Vj = &sram[tile_size*3];
    float *S = &sram[tile_size*4];
    for(int i=0; i<Tr; i++){
        for(int x=0; x<d; x++){
            Qi[tx*d + x] = Q[qkv_offset + i * tile_size + tx*d + x];
            Oi[tx*d + x] = 0;
        }
        __syncthreads(); 
        // 偏移到第几行
        float row_m_prev = -INFINITY;
        float row_l_prev = 0;
        float row_m_new, row_l_new;
        for(int j=0; j<Tc; j++){
            for(int x=0; x<d; x++){
                Kj[tx*d + x] = K[qkv_offset + j*tile_size + tx*d + x];
                Vj[tx*d + x] = V[qkv_offset + j*tile_size + tx*d + x];
            }
            
            float row_m = -INFINITY;
            #pragma unroll
            for(int y=0; y<Bc; y++){
                float sum = 0;
                for(int x=0; x<d; x++){
                    sum += Qi[tx*d+ x] * Kj[y*d+x];
                }
                sum *=softmax_scale;
                // S = [Bc, Bc]
                S[tx*Bc + y] = sum;
                row_m = max(row_m, sum);
            }
            row_m_new = max(row_m_prev, row_m);
            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l=0;
            for(int y=0; y<Bc; y++){
                S[tx*Bc + y] = __expf(S[tx*Bc + y] - row_m_new);
                row_l += S[tx*Bc + y];
            }
            row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

            // Write O, l, m to HBM
            #pragma unroll
            for(int x=0; x<d; x++){
                float pv = 0;
                 // [Bc, Bc] * [Bc, d], 这里是行乘以列，要注意Vj的索引
                for(int y=0; y<Bc; y++){
                    pv += S[Bc*tx + y] * Vj[y*d + x];
                }
                // 这里为什么要加上历史的Oi呢？ 
                // 因为历史的Oi 只是Qi和一个Bc大小的K相乘获得的结果S，后续还有很多个(Tc个) Bc大小的K要和Qi进行相乘，获得S_ij.这里的S_ij append到S_i的后面，形成Qi的完整Si(这里i代表一行)
                // 然后不同的tile的Vj去和S_ij进行相乘, 这里的j代表一个特定的Tc tile，大小为Bc.
                // 每一个tile的Vj都会得到一个Oij的结果，但是最终O要的结果是，整个S_i * 整个V_j的结果，所以要加上历史的Oi, 即历史的Oi为部分S_i * 部分V_j
                // [Bc, d] 
                Oi[tx*d + x] = __expf(row_m_prev - row_m_new) * Oi[tx*d + x] + pv;        
            }
            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }
        // 计算一行的O
        for(int x=0; x<d; x++){
            O[qkv_offset + i*tile_size + tx*d +x] = 
                Oi[tx*d + x] / row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward_V2(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    const int Bc = 32, Br = 32;
    const int bs = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int Tc = ceil((float) N / Bc), Tr = ceil((float) N / Br);

    const float softmax_scale = 1.0 / sqrt(d);
    auto O = torch::zeros_like(Q);
    auto m = torch::full({bs, nh, N}, -INFINITY);
    auto l = torch::zeros({bs, nh, N});
    torch::Device device(torch::kCUDA);
    l = l.to(device), m = m.to(device);

    //计算sram
    const int sram_size = (4 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("[Flash Attn v2] Max shared mem: %d, request shared mem: %d \\n", max_sram_size, sram_size);

    // 分配dim
    dim3 grid_dim(bs, nh);
    dim3 block_dim(Bc);
    forward_V2_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;

}

