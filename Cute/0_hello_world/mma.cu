
#include <cuda.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

using namespace cute;


void example_flash_mma(){
    // 比如这里的mma划分， flash attn为了将同一行的数据放在同一个warp内，方便在warp内循环计算softmax
    // EU layout 和 permutation把warp按 竖方向进行排列
    
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // 4 个warp全都在m方向， 则其shape 为 64x8x16
    static constexpr int kMmaEURepeatM = 4; // 4-M-warp
    static constexpr int kMmaEURepeatN = 1; // 1-N-warp
    static constexpr int kMmaEURepeatK = 1; // always 1

    static constexpr int kMmaPM = 64; // Perm-M
    static constexpr int kMmaPN = 32; // Perm-N       // column方向上同一个warp循环4次
    static constexpr int kMmaPK = 16; // always 16 when using 16x8x16 mma op

}

int main() {
    // 0. 构造 tiled mma
    // 调用tensorcore 完成一次m16n8k16的矩阵乘法 C(16*8) = A(16*16) * B(8*16)
    // c 为F32, A、B为F16, 累加器为F32
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // target 128*128*16
    // 具体线程分工： 16x8 = 128 => 128 / 32 = 4, 一个线程4个数据

    // 1. EU layout  
    // ** warp 排布 ** 
    // 这里为SM80， 故单位粒度为一个warp，执行MMA， 4个warp ==> 32x16x16
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    
    // 2. permutation ：指定一次threadblock能做的mma块大小
    using mma_atom_shape = mma_traits::Shape_MNK;   
    static constexpr int kMmaPM = 32; // 32    
    static constexpr int kMmaPN = 32; // 32    ** 这里每个warp在列方向上循环两次，才能得到32x32x16 ** 
    static constexpr int kMmaPK = 16; // 16

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    auto thr_mma = MMA{}.get_thread_slice(0);
   // print(thr_mma);
   // print(MMA{});
    MMA tiled_mma;

    print_latex(tiled_mma);
}