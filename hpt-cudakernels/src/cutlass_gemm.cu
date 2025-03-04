#include "cutlass/cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass/numeric_types.h"
#include "cutlass/cutlass/layout/matrix.h"

template <typename T>
struct CutlassTypes
{
    using ElementA = T;
    using ElementB = T;
    using ElementC = T;
    using ElementAccum = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

#if __CUDA_ARCH__ >= 800
    using ArchTag = cutlass::arch::Sm80;
#elif __CUDA_ARCH__ >= 750
    using ArchTag = cutlass::arch::Sm75;
#elif __CUDA_ARCH__ >= 700
    using ArchTag = cutlass::arch::Sm70;
#else
    using ArchTag = cutlass::arch::Sm70;
#endif

    using GemmConfig = typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt,
        ArchTag,
        ElementA,
        ElementB,
        ElementC,
        ElementAccum>;

    using GemmOp = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccum,
        GemmConfig>;
};

extern "C" void cutlass_gemm_f32(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_size)
{
    using Config = CutlassTypes<float>;
    using Gemm = typename Config::GemmOp;

    Gemm::Arguments args({M, N, K},
                         {A, K},
                         {B, N},
                         {C, N},
                         {C, N},
                         {1.0f, 0.0f});

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        return;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess)
    {
        return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess)
    {
        return;
    }
}