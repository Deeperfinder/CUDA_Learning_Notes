#include <torch/types.h>
#include <torch/extension.h>


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

// from hgemm.cu
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
// from hgemm_mma_stage_tn_cute.cu
void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);

// from hgemm_cublas.cu
void init_cublas_handle();
void destroy_cublas_handle();
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_stages_block_swizzle_tn_cute)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_naive_f16)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_sliced_k_f16)

  TORCH_BINDING_COMMON_EXTENSION(init_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tensor_op_nn)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tensor_op_tn)
}
