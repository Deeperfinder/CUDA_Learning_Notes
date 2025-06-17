#include <iostream>
#include <vector>
#include <cuda_runtime.h> // For CUDA API functions
#include <assert.h>       // For assert

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA Kernel for GEMV: y = A * x
// Each thread computes one element of y
__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    // Calculate global row index for this thread
    // Each thread computes one row of A multiplied by x
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

int main() {
    int M = 1024; // Number of rows in A (and elements in y)
    int N = 2048; // Number of columns in A (and elements in x)

    size_t size_A = M * N * sizeof(float);
    size_t size_x = N * sizeof(float);
    size_t size_y = M * sizeof(float);

    // 1. Host memory allocation and initialization
    std::vector<float> h_A(M * N);
    std::vector<float> h_x(N);
    std::vector<float> h_y(M); // To store results from device
    std::vector<float> h_y_ref(M); // To store reference results from CPU

    // Initialize A and x with some values
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = static_cast<float>(i * N + j) / 1000.0f;
        }
    }
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i) / 100.0f;
    }

    // 2. Device memory allocation
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_x, size_x));
    CUDA_CHECK(cudaMalloc(&d_y, size_y));

    // 3. Host to Device memory copy
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice));

    // 4. Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching GEMV kernel with M=" << M << ", N=" << N << std::endl;
    std::cout << "Grid Dim: " << blocksPerGrid << ", Block Dim: " << threadsPerBlock << std::endl;

    // 5. Kernel launch
    gemv_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, M, N);

    // 6. Synchronize and check for kernel errors
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // 7. Device to Host memory copy
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, size_y, cudaMemcpyDeviceToHost));

    // 8. CPU reference computation for verification
    std::cout << "Performing CPU reference computation..." << std::endl;
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += h_A[i * N + j] * h_x[j];
        }
        h_y_ref[i] = sum;
    }

    // 9. Verification
    bool passed = true;
    float tolerance = 1e-3f; // Adjust tolerance as needed
    for (int i = 0; i < M; ++i) {
        if (std::abs(h_y[i] - h_y_ref[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i << ": CUDA=" << h_y[i] << ", CPU=" << h_y_ref[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // 10. Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
