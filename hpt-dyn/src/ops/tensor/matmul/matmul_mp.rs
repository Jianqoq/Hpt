use gemm_common::cache::DivCeil;
use hpt_common::Pointer;
use hpt_traits::tensor::CommonBounds;

use super::{microkernel_trait::MatmulMicroKernel, template::matmul_mp, utils::kernel_params};
use hpt_types::{dtype::TypeCommon, traits::VecTrait};

#[inline(always)]
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
        
        |_, _, _| T::ZERO,
        |_, _, _| T::Vec::splat(T::ZERO),
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back,
    );
}

#[duplicate::duplicate_item(
    func_name half_type half_str;
    [f16_matmul_mp_no_block_info] [half::f16] ["f16"];
    [bf16_matmul_mp_no_block_info] [half::bf16] ["bf16"];
)]
pub(crate) fn func_name<T, IM>(
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
    
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
{
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half_type as TypeCommon>::Vec;
    assert_eq!(T::STR, half_str);
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
            let val = val as *const T as *const half_type;
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
            let im = im as *mut T as *mut half_type;
            unsafe { *im = half_type::from_f32(*val) };
        },
    )
}
