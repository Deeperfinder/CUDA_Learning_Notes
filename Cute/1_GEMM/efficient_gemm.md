优化方法：
# 计算
```c++
  // 1. 硬件指令原子 (Hardware Instruction Atom)
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  // 2. 线程内 MMA 重复 (Per-Thread MMA Repetition)
  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  // 3. 计算 Warp 级分片大小 (Calculating Warp-level Partition Size)
  using mma_atom_shape = mma_traits::Shape_MNK; // e.g., Shape<_16, _8, _16>
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 1 * 2 * 16 = 32
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2 * 2 * 8  = 32
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 1 * 1 * 16 = 16

  // 4. 定义 TiledMMA 的两个关键 Layout
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
      // Layout<Shape<_2, _2, _1>>
      
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>; 
      // Tile<Shape<_32, _32, _16>> (This is a Layout)

  // 5. 最终组装 TiledMMA
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  ```
1. 硬件指令原子 (mma_atom)

* mma_op = SM80_16x8x16_F16F16F16F16_TN;: 这定义了我们要使用的底层硬件指令。这是 NVIDIA Ampere (SM80) 架构上的 __mma.sync__ 指令。
    * 16x8x16: 指令处理的矩阵分块大小为 M=16, N=8, K=16。
    * F16F16F16F16: 输入矩阵 A, B 是 FP16，累加和输出矩阵 C 是 FP16。
    * TN: A 矩阵为行优先，B矩阵为列优先，(blas 中约定normal矩阵为列优先，T表示transpose, 即对列优先的矩阵进行转置则为行优先)
    * mma_atom: 这是 CuTe 对该硬件指令的抽象，是整个计算中不可再分的最小单元。它的 "形状" (Shape) 是 Shape<_16, _8, _16>。
2. 线程内 MMA 重复 (kMmaEURepeat* 和 MMA_EU_RepeatT)
* kMmaEURepeatM = 2, kMmaEURepeatN = 2: 这定义了单个线程在其寄存器中要累积的 mma_atom 的数量和排列方式。这里定义了一个 2x2 的网格。
* 物理意义: 一个线程不是只执行一次 16x8 的 MMA 计算，而是执行 2 * 2 = 4 次。这 4 次计算的结果在线程的寄存器中形成一个更大的分块。
* 线程计算的 C 矩阵分块大小:
    * M维度: kMmaEURepeatM * mma_atom::M = $2 * 16 = 32  $
    * N维度: kMmaEURepeatN * mma_atom::N = $2 * 8 = 16 $  <br>
所以，每个线程负责计算并维护一个 32x16 大小的 C 矩阵分块。
3. Warp 级协作 (MMA_P_T 和 make_tiled_mma)

这是最关键也最复杂的部分。单个线程只能计算一个 32x16 的小块，但一个 Warp (32个线程) 需要协同计算一个更大的 WARP_TILE。make_tiled_mma 将前面定义的 mma_atom 和 MMA_EU_RepeatT 组合起来，定义了这种协作模式。

在 Ampere 架构上，执行 `mma.sync` 指令的 Warp (32个线程) 通常被划分为 __4__ 个线程组 (thread group)，每组 8 个线程。这 8 个线程协同完成一组 MMA 计算。

* make_tiled_mma 的作用: 它创建一个 TiledMMA 对象，这个对象描述了一个 8 线程的组如何协作。
* 协作结果: 这 8 个线程共同计算的 WARP_TILE 是多大呢？
    * 每个线程贡献一个 32x16 的分块，共 32 * 16 = 512 个元素。
    * 8 个线程总共贡献 8 * 512 = 4096 个元素。
    * 这 4096 个元素通常被组织成一个 64x64 的 WARP_TILE (64 * 64 = 4096)。
* MMA 类型的含义:
`using MMA = decltype(make_tiled_mma(...));`
这个 MMA 类型封装了上述所有信息。当你创建一个 MMA 类型的变量时，它就代表了一个由 8 个线程协作计算的 64x64 大小的 WARP_TILE。它内部知道，这个 64x64 的瓦片是由 8 个线程分工的，每个线程负责其中一个 32x16 的部分，而每个 32x16 的部分又是由 4 个 16x8 的 `mma_atom` 组成的。
4. kMmaPM 和 kMmaPN 的作用

这些复杂的 constexpr 计算是用来指导 `make_tiled_mma` 如何构建内部映射的，它们本身并不是最终的 WARP_TILE 尺寸，但参与了其构建。这是一种高度参数化的方式，允许 CUTLASS 通过改变 `kMmaEURepeat*` 和一些乘数（这里的1和2）来灵活地生成不同的 MMA 调度策略。

这里定义具体的数据分配和计算策略：<br>
**硬件基础**: `16x8x16 MMA` 指令。<br>
**线程级**: 每个线程计算一个 **32x16** 的 C 矩阵分块。<br>
**Warp级 (线程组)**: 每 8 个线程组成一个小组，协同计算一个 64x64 的 C 矩阵 WARP_TILE。<br>
**CTA级**: 如果 kTileM=128, kTileN=128，那么一个 128 线程的块 (4个Warp) 会被组织起来。每个 Warp (包含 4 个 8-线程组) 可能会负责 CTA_TILE 的一部分，例如，4 个 Warp 可能形成一个 2x2 的网格，每个 Warp 负责一个 64x64 的区域，共同覆盖 128x128 的 CTA_TILE。

# 访存
## 全局内存到共享内存异步拷贝(__DMA__) <br>
在cute中，针对全局内存到共享内存。选择cute已经定义好的抽象能力即可。`SM80_CP_ASYNC_CACHEGLOBAL`COPY_Operation, 该指令可以实现全局内存到共享内存的异步拷贝。同时 __CACHEGLOBAL__ 指定了数据只在L2做Cache，对L1则做bypass，于是我们可以形成如下主机端代码：
```c++
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;
```
和MMA时的make_tiled_mma类似，Copy抽象提供了make_tile_copy能力，其通过制定线程和数据的重复方法将Atom能力扩展到块状能力。数据拷贝时可以区分AB矩阵的不同拷贝方法，此处选用同样的Copy能力。设备端代码如下：
```c++
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)
```
Copy时先将**TileCopy**通过指定线程号得到线程级的Copy能力抽象**ThrCopy**。也和图2中的MMA划分类似，ThrCopy抽象提供了`partition_S/D`函数，其实现将大块的矩阵划分到线程维度上。`CPY` 维度代表该线程需要做的数据大小，`CPY_M、CPY_K`表示在给定的划分的**Tile**需要在row 和 col方向重复的次数，如果被划分的Tile维度大于2，多出的维度附加到(, M, K)维度的后面
## 共享内存到寄存器的ldmatrix指令
Host 端代码：
```c++
//shared memory to register copy
using s2r_copy_op = SM75_U32x4_LDSM_N;
using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

using S2RCopyAtomA = s2r_copy_atom;
using S2RCopyAtomB = s2r_copy_atom;
```
Device 端代码:
```c++
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)
```
主机端选择ldmatrix指令的x4模式，形成Atom抽象，设备端通过`make_tiled_copy_A`函数借助tiled_mma抽象出共享内存到寄存的TileCopy。   
与前面的G2S TileCopy不同，这里直接使用`tiled_mma`的信息形成块状拷贝。对于它作为目标的Copy而言，`tiled_mma`自然也是精准描述这部分的数据的，所以就不需要用户额外指定Copy_Atom到TileCopy的信息，而是直接从**MMA**能力中获得。由于**MMA**已经声明了寄存器的存储空间，这里直接对其进行线程级别的小块retile即可，不再是大块到小块的partition。

# 算法高效  
这里主要包括两个部分:  
1. 分块  
定义了分块的大小和设备端如何通过Tensor抽象和local_tile将矩阵进行分块
```cpp
// Host
static constexpr int kTileM = kTileM_;
static constexpr int kTilnN = kTileN_;
static constexpr int kTilnK = kTilnK_;
static constexpr int kStage = kStage_;
// Device
// slice the tensor to small one which is used for current thread block
Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
Tensor gB = local_tile(B, make_tile(Int<kTileN>{}，Int<kTileK>{}), make_coord(ix, _));
Tensor gD = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
// shared memory
auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});
```

2. 流水线
为了做multi-stage 流水线，需要在共享内存分配时指定流水线级数，同时设备端需要做必要的数据加载和计算重叠方案。主机端代码如下
```cpp
static constexpr int kShmLoadSwizzleM = 3;
static constexpr int kShmLoadSwizzleN = 3;
static constexpr int kShmLoadSwizzleK = 3;

using SmemLayoutAtom = decltype(composition(
    Swizzle<kshmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
    make_layout(make_shape(Int<8>{}, Int<KTileK>{}),
                make_stride(Int<KTileK>{}, Int<1>{})))); // Swizzle<3, 3, 3> Layout

using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kStage>{}))
);
using SmemLayoutB = decltype(
    tile_to_shape(SmemLayoutAtom{}, make_shape(Int<KtileN>{}, Int<kTileK>{}, Int<kStage>{}));
)
```
其定义了共享内存的Layout， 其中Swizzle用来避免bank conflict, 其中kStage表示流水线的级数.

    
