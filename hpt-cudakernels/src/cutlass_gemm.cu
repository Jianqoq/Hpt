#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

void gen_rand_data(float *data, int n);

template <int kTileM, int kTileN, int kTileK, typename TiledMMA>
__device__ void gemm_simple(float *Cptr, const float *Aptr, const float *Bptr, int m, int n, int k)
{

    using namespace cute;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    //  gA(kTileM, kTileK, num_tile_k)
    //  gB(kTileN, kTileK, num_tile_k)
    //  gC(kTileM, kTileN)

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)

    clear(tCrC);

    int num_tile_k = size<2>(gA);
#pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile)
    {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}

extern "C" __global__ void gemm_simple_kernel(float *Cptr, const float *Aptr, const float *Bptr, int m, int n, int k)
{
    using namespace cute;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        make_layout(Shape<_1, _2, _1>{})));
    gemm_simple<16, 16, 16, MMA>(Cptr, Aptr, Bptr, m, n, k);
}
