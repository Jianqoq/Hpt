use gemm_common::cache::DivCeil;
use hpt_common::{Pointer, error::base::TensorError};
use hpt_traits::tensor::CommonBounds;

use crate::Device;

use super::{
    microkernel_trait::MatmulMicroKernel,
    template::matmul_mp,
    utils::{PrePackedRhs, kernel_params},
};
use hpt_types::{dtype::ToDType, traits::VecTrait};

#[allow(unused)]
pub(crate) fn matmul_mp_no_block_info_prepack_rhs<T, IM>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    device: Device,
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(&mut IM, &T),
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds + ToDType,
{
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), true);
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.msrv_next_multiple_of(nr);
    }
    super::template::prepack_mp_b::<T, IM>(
        b,
        m,
        n,
        k,
        ldb,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
        device,
        pack_vec,
        pack_vec_exceed,
        pack_zero,
    )
}

#[allow(unused)]
pub(crate) fn matmul_mp_no_block_info<T, IM>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    prepack_rhs: Option<PrePackedRhs>,
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
    cast_back: fn(&mut T, &IM),
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), true);
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.msrv_next_multiple_of(nr);
    }
    matmul_mp::<T, IM, _, _>(
        a,
        b,
        out,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
        prepack_rhs,
        |_, _, _| T::ZERO,
        |_, _, _| T::Vec::splat(T::ZERO),
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back,
    );
}

#[cfg(feature = "f16")]
pub(crate) fn f16_matmul_mp_no_block_info_prepack_rhs<T, IM>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    device: Device,
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    use hpt_types::dtype::TypeCommon;
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::f16 as TypeCommon>::Vec;
    assert_eq!(T::STR, "f16");
    assert_eq!(IM::STR, "f32");
    matmul_mp_no_block_info_prepack_rhs::<T, f32>(
        b,
        m,
        n,
        k,
        ldb,
        rhs_col_stride,
        num_threads,
        device,
        |packed_b, b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let b = b as *const F16Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |im, val| {
            let val = val as *const T as *const half::f16;
            *im = unsafe { val.read().to_f32() };
        },
    )
}

#[cfg(feature = "f16")]
pub(crate) fn f16_matmul_mp_no_block_info<T, IM>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    prepack_rhs: Option<PrePackedRhs>,
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    use hpt_types::dtype::TypeCommon;
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::f16 as TypeCommon>::Vec;
    assert_eq!(T::STR, "f16");
    assert_eq!(IM::STR, "f32");
    matmul_mp_no_block_info::<T, f32>(
        a,
        b,
        out,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        num_threads,
        prepack_rhs,
        |packed_b, b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let b = b as *const F16Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |im, val| {
            let val = val as *const T as *const half::f16;
            *im = unsafe { val.read().to_f32() };
        },
        |im, val| {
            let im = im as *mut F16Vec;
            let val = val as *const F32Vec;
            let vec0 = unsafe { val.read() };
            let vec1 = unsafe { val.add(1).read() };
            unsafe { im.write_unaligned(F16Vec::from_2_f32vec([vec0, vec1])) };
        },
        |im, val| {
            let im = im as *mut T as *mut half::f16;
            unsafe { *im = half::f16::from_f32(*val) };
        },
    )
}

#[cfg(feature = "bf16")]
pub(crate) fn bf16_matmul_mp_no_block_info_prepack_rhs<T, IM>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    device: Device,
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    use hpt_types::dtype::TypeCommon;
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::bf16 as TypeCommon>::Vec;
    assert_eq!(T::STR, "bf16");
    assert_eq!(IM::STR, "f32");
    matmul_mp_no_block_info_prepack_rhs::<T, f32>(
        b,
        m,
        n,
        k,
        ldb,
        rhs_col_stride,
        num_threads,
        device,
        |packed_b, b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let b = b as *const F16Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |im, val| {
            let val = val as *const T as *const half::bf16;
            *im = unsafe { val.read().to_f32() };
        },
    )
}

#[cfg(feature = "bf16")]
pub(crate) fn bf16_matmul_mp_no_block_info<T, IM>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    prepack_rhs: Option<PrePackedRhs>,
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    use hpt_types::dtype::TypeCommon;
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::bf16 as TypeCommon>::Vec;
    assert_eq!(T::STR, "bf16");
    assert_eq!(IM::STR, "f32");
    matmul_mp_no_block_info::<T, f32>(
        a,
        b,
        out,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        num_threads,
        prepack_rhs,
        |packed_b, b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let b = b as *const F16Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |im, val| {
            let val = val as *const T as *const half::bf16;
            *im = unsafe { val.read().to_f32() };
        },
        |im, val| {
            let im = im as *mut F16Vec;
            let val = val as *const F32Vec;
            let vec0 = unsafe { val.read() };
            let vec1 = unsafe { val.add(1).read() };
            unsafe { im.write_unaligned(F16Vec::from_2_f32vec([vec0, vec1])) };
        },
        |im, val| {
            let im = im as *mut T as *mut half::bf16;
            unsafe { *im = half::bf16::from_f32(*val) };
        },
    )
}
