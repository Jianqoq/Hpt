use std::cmp::min;

use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::ops::creation::TensorCreator;
use hpt_types::type_promote::NormalOut;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use crate::{Tensor, ALIGN};
use hpt_common::shape::shape_utils::compare_and_pad_shapes;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;

pub(crate) fn gemm_prepare<T, const DEVICE: usize, A>(
    lhs: &_Tensor<T, Cpu, DEVICE, A>,
    rhs: &_Tensor<T, Cpu, DEVICE, A>,
    out: Option<_Tensor<T, Cpu, DEVICE, A>>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(
                &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
                &out.layout(),
            )?;
            Ok(out)
        } else {
            _Tensor::<T, Cpu, DEVICE, A>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])
        };
        res
    } else {
        let (longer_shape, padded_short_shape) = compare_and_pad_shapes(&lhs.shape(), &rhs.shape());
        let a_shape;
        let b_shape;
        if lhs.shape().len() > rhs.shape().len() {
            a_shape = longer_shape;
            b_shape = padded_short_shape;
        } else {
            a_shape = padded_short_shape;
            b_shape = longer_shape;
        }
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let mut res_shape =
            predict_broadcast_shape(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?
                .to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            Ok(out)
        } else {
            _Tensor::<T, Cpu, DEVICE, A>::zeros(res_shape)
        };
        res
    }
}

pub(crate) fn gemm2d<
    T,
    const MR: usize,
    const NR: usize,
    const MC: usize,
    const NC: usize,
    const KC: usize,
    const NR_DIV_LANE: usize,
>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    stride: i64,
) where
    T: CommonBounds,
{
    let num_mr_blocks = (MC + MR - 1) / MR;
    let num_nr_blocks = (NC + NR - 1) / NR;
    let packed_a_layout = std::alloc::Layout::from_size_align(
        num_mr_blocks * MR * KC * std::mem::size_of::<T>(),
        ALIGN,
    )
    .expect("layout create failed");
    let packed_b_layout = std::alloc::Layout::from_size_align(
        num_nr_blocks * NR * KC * std::mem::size_of::<T>(),
        ALIGN,
    )
    .expect("layout create failed");
    let allocate_buffer = || {
        let packed_a_origin = unsafe { std::alloc::alloc(packed_a_layout) };
        let packed_b_origin = unsafe { std::alloc::alloc(packed_b_layout) };
        if packed_a_origin == std::ptr::null_mut() || packed_b_origin == std::ptr::null_mut() {
            panic!("alloc failed");
        }
        #[cfg(feature = "bound_check")]
        let packed_a = Pointer::new(
            packed_a_origin as *mut T,
            (packed_a_layout.size() / std::mem::size_of::<T>()) as i64,
        );
        #[cfg(feature = "bound_check")]
        let packed_b = Pointer::new(
            packed_b_origin as *mut T,
            (packed_b_layout.size() / std::mem::size_of::<T>()) as i64,
        );
        #[cfg(not(feature = "bound_check"))]
        let packed_a = Pointer::new(packed_a_origin as *mut T);
        #[cfg(not(feature = "bound_check"))]
        let packed_b = Pointer::new(packed_b_origin as *mut T);
        (packed_a, packed_b)
    };
    let n_blocks = n.div_ceil(NC);
    let num_threads = n_blocks.min(rayon::current_num_threads());
    // println!("num_threads: {}", num_threads);
    let blocks_per_thread = n_blocks.div_ceil(num_threads);
    let buffers = (0..num_threads)
        .map(|_| allocate_buffer())
        .collect::<Vec<_>>();
    for p in (0..k).step_by(KC) {
        let pb = min(KC, k - p);

        buffers
            .par_iter()
            .enumerate()
            .for_each(|(tid, (packed_a, packed_b))| {
                let start_block = tid * blocks_per_thread;
                let end_block = min((tid + 1) * blocks_per_thread, n_blocks);
                for block in start_block..end_block {
                    let j = block * NC;
                    let jb = min(NC, n - j);
                    pack_b::<T, NR, KC>(
                        b.clone() + (p as i64 * ldb + j as i64),
                        packed_b.clone(),
                        ldb,
                        jb,
                        pb,
                    );

                    for i in (0..m).step_by(MC) {
                        let ib = min(MC, m - i);
                        pack_a::<T, MR, KC>(
                            a.clone() + i as i64 * lda + p as i64,
                            packed_a.clone(),
                            lda,
                            stride,
                            ib,
                            pb,
                        );
                        micro_kernel::<T, MR, NR, KC, NR_DIV_LANE>(
                            packed_a.clone(),
                            packed_b.clone(),
                            out.clone() + i as i64 * ldc + j as i64,
                            ib,
                            jb,
                            ldc,
                        );
                    }
                }
            });
    }

    for (packed_a, packed_b) in buffers {
        unsafe {
            std::alloc::dealloc(packed_a.ptr as *mut u8, packed_a_layout);
            std::alloc::dealloc(packed_b.ptr as *mut u8, packed_b_layout);
        }
    }
}

pub(crate) fn pack_a<T, const MR: usize, const KC: usize>(
    a: Pointer<T>,
    mut packed_a: Pointer<T>,
    lda: i64,
    stride: i64,
    mc: usize,
    kc: usize,
) where
    T: CommonBounds,
{
    for i in (0..mc).step_by(MR) {
        let mr = MR.min(mc - i);
        for p in 0..kc as i64 {
            for ii in 0..mr as i64 {
                let i = i as i64 + ii;
                *packed_a = a[i * lda + p * stride];
                packed_a += 1i64;
            }
            for _ in mr..MR {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
        for _ in kc..KC {
            for _ in 0..MR {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

pub(crate) fn pack_b<T, const NR: usize, const KC: usize>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    nc: usize,
    kc: usize,
) where
    T: CommonBounds,
{
    for j in (0..nc).step_by(NR) {
        let nr = NR.min(nc - j);
        for p in 0..kc as i64 {
            for jj in 0..nr as i64 {
                let j = j as i64 + jj;
                *packed_b = b[p * ldb + j];
                packed_b += 1i64;
            }
            for _ in nr..NR {
                *packed_b = T::ZERO;
                packed_b += 1i64;
            }
        }
        for _ in kc..KC {
            for _ in 0..NR {
                *packed_b = T::ZERO;
                packed_b += 1i64;
            }
        }
    }
}

pub(crate) fn micro_kernel<
    T,
    const MR: usize,
    const NR: usize,
    const KC: usize,
    const NR_DIV_LANE: usize,
>(
    mut packed_a: Pointer<T>,
    mut packed_b: Pointer<T>,
    c: Pointer<T>,
    mc: usize,
    nc: usize,
    ldc: i64,
) where
    T: CommonBounds,
{
    let packed_b_cpy = packed_b.clone();
    for i in (0..mc).step_by(MR) {
        let packed_a_cpy = packed_a.clone();
        packed_b = packed_b_cpy.clone();

        for j in (0..nc).step_by(NR) {
            let mut c_local = [[T::Vec::splat(T::ZERO); NR_DIV_LANE]; MR];
            packed_a = packed_a_cpy.clone();
            for _ in 0..KC {
                let b_vec = unsafe { T::Vec::from_ptr(packed_b.ptr as *const T) };
                let b_vec1 =
                    unsafe { T::Vec::from_ptr(packed_b.ptr.add(T::Vec::SIZE) as *const T) };
                let a_vec = T::Vec::splat(packed_a[0i64]);
                let a_vec1 = T::Vec::splat(packed_a[1i64]);
                let a_vec2 = T::Vec::splat(packed_a[2i64]);
                let a_vec3 = T::Vec::splat(packed_a[3i64]);
                c_local[0][0] = a_vec._mul_add(b_vec, c_local[0][0]);
                c_local[0][1] = a_vec._mul_add(b_vec1, c_local[0][1]);
                c_local[1][0] = a_vec1._mul_add(b_vec, c_local[1][0]);
                c_local[1][1] = a_vec1._mul_add(b_vec1, c_local[1][1]);
                c_local[2][0] = a_vec2._mul_add(b_vec, c_local[2][0]);
                c_local[2][1] = a_vec2._mul_add(b_vec1, c_local[2][1]);
                c_local[3][0] = a_vec3._mul_add(b_vec, c_local[3][0]);
                c_local[3][1] = a_vec3._mul_add(b_vec1, c_local[3][1]);
                packed_b += NR as i64;
                packed_a += MR as i64;
            }
            for ii in 0..MR as i64 {
                for jj in 0..NR_DIV_LANE as i64 {
                    let res_idx = (i as i64 + ii) * ldc + (j as i64 + jj * T::Vec::SIZE as i64);
                    let res_ptr = unsafe { c.ptr.offset(res_idx as isize) } as *mut T::Vec;
                    unsafe {
                        res_ptr.write_unaligned(
                            res_ptr
                                .read_unaligned()
                                ._add(c_local[ii as usize][jj as usize]),
                        );
                    };
                }
            }
        }
    }
}

/// gemm
pub fn gemm<T, const DEVICE: usize, A>(
    a: &Tensor<T, Cpu, DEVICE, A>,
    b: &Tensor<T, Cpu, DEVICE, A>,
    out: Option<Tensor<T, Cpu, DEVICE, A>>,
) -> Result<Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    let c = gemm_prepare(&a.inner, &b.inner, out.map(|t| t.inner.as_ref().clone()))?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;
    let lda = a.shape()[1] as i64;
    let ldb = b.shape()[1] as i64;
    let ldc = c.shape()[1] as i64;
    let stride = 1;
    gemm2d::<T, 4, 16, 32, 32, 32, { 16 / 8 }>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        stride,
    );
    Ok(c.into())
}
