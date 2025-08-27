#include <cuda.h>

template<typename T>
void gen_rand_data(T *data, int n){
    for(int i=0; i<n; i++){
        float v = (rand() % 200 -100) * 0.01;
        data[i] = v;
    }
}


template<typename T>
float perf_gemm_swizzle(void (*gpu_hgemm)(T*, T*, T*, int, int, int),
                  int m, int n, int k, int inner_repeat, int warm_up = 2){
    // 初始化data
    // 传入指针以便修改Aptr的内存地址

    T *Cptr;
    T *Aptr;
    T *Bptr;

    T *Cptr_host;
    T *Aptr_host;
    T *Bptr_host;
    // 这里也可以使用cudaMalloc直接赋值
    // cudaMalloc(&Cptr, size_a);
    cudaMalloc(&Aptr, sizeof(T) * m * k);
    cudaMalloc(&Bptr, sizeof(T) * n * k);
    cudaMalloc(&Cptr, sizeof(T) * m * n);

    Aptr_host = (T*)malloc(sizeof(T) * m * k);
    Bptr_host = (T*)malloc(sizeof(T) * n * k);
    Cptr_host = (T*)malloc(sizeof(T) * m * n);

    gen_rand_data(Aptr_host, m*k);
    gen_rand_data(Bptr_host, n*k);

    cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice); 

    // warmup
    for(int i=0; i < warm_up; i++){
        gpu_hgemm(Cptr, Aptr, Bptr, m, n, k);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i =0; i < inner_repeat; i++){
        gpu_hgemm(Cptr, Aptr, Bptr, m, n, k);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    float msec, sec;
    cudaEventElapsedTime(&msec, start, stop);
    sec = msec / 1000.0 / inner_repeat;

    cudaFree(Aptr);
    cudaFree(Bptr);
    cudaFree(Cptr);
    free(Aptr_host);
    free(Bptr_host);
    free(Cptr_host);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return sec;
}


