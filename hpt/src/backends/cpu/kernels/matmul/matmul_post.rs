use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use gemm_common::cache::DivCeil;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;

use super::common::matmul_prepare;
use super::microkernel_trait::MatmulMicroKernel;
use super::utils::kernel_params;

/// single batch matmul template no block info
///
/// # Arguments
///
/// * `a`: lhs shape `(m, k)`
/// * `b`: rhs shape `(k, n)`
/// * `out`: output shape `(m, n)`
/// * `m`: rows of lhs
/// * `n`: cols of rhs
/// * `k`: cols of lhs
/// * `lda`: `lhs.strides[a.ndim() - 2]`
/// * `ldb`: `rhs.strides[r.ndim() - 2]`
/// * `ldc`: `out.shape[out.ndim() - 1]`
/// * `lhs_col_stride`: `lhs.strides[a.ndim() - 1]`
/// * `rhs_col_stride`: `rhs.strides[b.ndim() - 1]`
/// * `post_op`: post operation
/// * `post_op_vec`: post operation for vector
/// * `num_threads`: number of threads
#[inline]
pub fn matmul_post_template_no_block_info<T, F1, F2>(
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
    post_op: F1,
    post_op_vec: F2,
    num_threads: usize,
) where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    let nr = T::get_max_nr() * T::Vec::SIZE;
    let mr = T::get_max_mr();
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if param.nc == 0 {
        param.nc = n.msrv_next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    super::template::matmul_post::<T, F1, F2>(
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
        do_lhs_pack,
        num_threads,
        post_op,
        post_op_vec,
    );
}

/// matmul
pub(crate) fn matmul_post<T, const DEVICE: usize, A, F1, F2>(
    a: &_Tensor<T, Cpu, DEVICE, A>,
    b: &_Tensor<T, Cpu, DEVICE, A>,
    out: Option<_Tensor<T, Cpu, DEVICE, A>>,
    post_op: F1,
    post_op_vec: F2,
    num_threads: usize,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    let c = matmul_prepare(&a, &b, out)?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;
    matmul_post_template_no_block_info::<T, F1, F2>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        a.strides()[a.ndim() - 2],
        b.strides()[b.ndim() - 2],
        c.strides()[c.ndim() - 2] as i64,
        a.strides()[a.ndim() - 1],
        b.strides()[b.ndim() - 1],
        post_op,
        post_op_vec,
        num_threads,
    );
    Ok(c.into())
}
