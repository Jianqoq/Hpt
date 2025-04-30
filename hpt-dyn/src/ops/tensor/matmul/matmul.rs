use gemm_common::cache::DivCeil;
use hpt_common::{
    Pointer,
    error::{base::TensorError, shape::ShapeError},
    shape::{
        shape::Shape,
        shape_utils::{compare_and_pad_shapes, mt_intervals, predict_broadcast_shape},
    },
    strides::strides_utils::preprocess_strides,
};
use hpt_traits::tensor::{CommonBounds, TensorInfo};

use crate::{THREAD_POOL, Tensor, current_num_threads};

use super::{
    common::matmul_prepare,
    matmul_mp::{bf16_matmul_mp_no_block_info, f16_matmul_mp_no_block_info},
    matmul_mp_post::{bf16_matmul_mp_post_no_block_info, f16_matmul_mp_post_no_block_info},
    matmul_post::matmul_post_template_no_block_info,
    microkernel_trait::MatmulMicroKernel,
    utils::kernel_params,
};
use hpt_types::{dtype::TypeCommon, traits::VecTrait, type_promote::NormalOut};

use half::{bf16, f16};

#[inline]
fn matmul_template_no_block_info<T>(
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
    super::template::matmul::<T, _, _>(
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
        |_, _, _| T::ZERO,
        |_, _, _| T::Vec::splat(T::ZERO),
    );
}

fn matmul<T, F1, F2>(
    lhs: Pointer<T>,
    rhs: Pointer<T>,
    out: Pointer<T>,
    lhs_strides: &[i64],
    rhs_strides: &[i64],
    out_strides: &[i64],
    lhs_shape: &[i64],
    rhs_shape: &[i64],
    out_shape: &[i64],
    threads: usize,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    let lhs_cs = lhs_strides[lhs_strides.len() - 1];
    let lhs_rs = lhs_strides[lhs_strides.len() - 2];
    let dst_cs = out_strides[out_strides.len() - 2];
    let rhs_cs = rhs_strides[rhs_strides.len() - 1];
    let rhs_rs = rhs_strides[rhs_strides.len() - 2];
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = out_shape[out_shape.len() - 2] as usize;
    match (post_op, post_op_vec) {
        (None, None) => match T::STR {
            "f16" => {
                if cfg!(target_feature = "neon") && cfg!(target_feature = "fp16") {
                    matmul_template_no_block_info::<f16>(
                        lhs.cast::<f16>(),
                        rhs.cast::<f16>(),
                        out.cast::<f16>(),
                        m,
                        n,
                        k,
                        lhs_rs,
                        rhs_rs,
                        dst_cs,
                        lhs_cs,
                        rhs_cs,
                        threads,
                    )
                } else {
                    f16_matmul_mp_no_block_info::<f16, f32>(
                        lhs.cast::<f16>(),
                        rhs.cast::<f16>(),
                        out.cast::<f16>(),
                        m,
                        n,
                        k,
                        lhs_rs,
                        rhs_rs,
                        dst_cs,
                        lhs_cs,
                        rhs_cs,
                        threads,
                    )
                }
            }
            "bf16" => bf16_matmul_mp_no_block_info::<bf16, f32>(
                lhs.cast::<bf16>(),
                rhs.cast::<bf16>(),
                out.cast::<bf16>(),
                m,
                n,
                k,
                lhs_rs,
                rhs_rs,
                dst_cs,
                lhs_cs,
                rhs_cs,
                threads,
            ),
            _ => matmul_template_no_block_info(
                lhs, rhs, out, m, n, k, lhs_rs, rhs_rs, dst_cs, lhs_cs, rhs_cs, threads,
            ),
        },
        (Some(post_op), Some(post_op_vec)) => match T::STR {
            "f16" => {
                if cfg!(target_feature = "neon") && cfg!(target_feature = "fp16") {
                    matmul_post_template_no_block_info(
                        lhs,
                        rhs,
                        out,
                        m,
                        n,
                        k,
                        lhs_rs,
                        rhs_rs,
                        dst_cs,
                        lhs_cs,
                        rhs_cs,
                        post_op,
                        post_op_vec,
                        threads,
                    )
                } else {
                    f16_matmul_mp_post_no_block_info::<T, f32, _, _>(
                        lhs,
                        rhs,
                        out,
                        m,
                        n,
                        k,
                        lhs_rs,
                        rhs_rs,
                        dst_cs,
                        lhs_cs,
                        rhs_cs,
                        threads,
                        post_op,
                        post_op_vec,
                    )
                }
            }
            "bf16" => bf16_matmul_mp_post_no_block_info::<T, f32, _, _>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                lhs_rs,
                rhs_rs,
                dst_cs,
                lhs_cs,
                rhs_cs,
                threads,
                post_op,
                post_op_vec,
            ),
            _ => matmul_post_template_no_block_info(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                lhs_rs,
                rhs_rs,
                dst_cs,
                lhs_cs,
                rhs_cs,
                post_op,
                post_op_vec,
                threads,
            ),
        },
        _ => panic!("post_op and post_op_vec must be both Some or both None"),
    }
}

#[track_caller]
pub(crate) fn matmul_with_out<T, F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    out: Option<Tensor>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) -> std::result::Result<Tensor, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        let c = matmul_prepare(&lhs, &rhs, out)?;
        matmul(
            lhs.data.cast::<T>(),
            rhs.data.cast::<T>(),
            c.data.cast::<T>(),
            lhs.strides(),
            rhs.strides(),
            c.strides(),
            lhs.shape(),
            rhs.shape(),
            c.shape(),
            current_num_threads(),
            post_op,
            post_op_vec,
        );
        Ok(c)
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
        let mut iterate_shape = res_shape.clone();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            out
        } else {
            Tensor::empty(&res_shape, lhs.dtype, lhs.device.clone())?
        };
        let a_strides = preprocess_strides(&a_shape, &lhs.strides());
        let b_strides = preprocess_strides(&b_shape, &rhs.strides());
        let batch = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
        let res_inner_matrix_size = (res.shape()[res.shape().len() - 2] as usize)
            * (res.shape()[res.shape().len() - 1] as usize);
        iterate_shape.iter_mut().for_each(|x| {
            *x -= 1;
        });
        let mut a_ptr = lhs.data.clone();
        let mut b_ptr = rhs.data.clone();
        let mut res_ptr = res.data.clone();
        let a_sizeof = lhs.dtype.sizeof();
        let b_sizeof = rhs.dtype.sizeof();
        let res_sizeof = res.dtype.sizeof();
        let rayon_num_threads = THREAD_POOL.with_borrow(|pool| pool.num_threads());
        let num_threads = batch.min(rayon_num_threads);
        let num_threads_each = if batch < rayon_num_threads {
            let vec = mt_intervals(rayon_num_threads, batch);
            vec.iter().map(|x| x.1 - x.0).collect::<Vec<usize>>()
        } else {
            vec![1; rayon_num_threads]
        };
        let intervals = mt_intervals(batch, num_threads);
        let mut res_ptrs = Vec::with_capacity(num_threads);
        let mut a_ptrs = Vec::with_capacity(num_threads);
        let mut b_ptrs = Vec::with_capacity(num_threads);
        let mut prgs = Vec::with_capacity(num_threads);
        let mut amount = 0;
        for i in 0..num_threads {
            let (start, end) = intervals[i];
            res_ptrs.push(res_ptr.clone());
            res_ptr.add((end - start) * res_inner_matrix_size * res_sizeof);
            let mut prg = vec![0i64; iterate_shape.len()];
            let mut amount_cpy = amount as i64;
            for j in (0..=iterate_shape.len() - 1).rev() {
                prg[j] = amount_cpy % (iterate_shape[j] + 1);
                amount_cpy /= iterate_shape[j] + 1;
                a_ptr += prg[j] * a_strides[j] * (a_sizeof as i64);
                b_ptr += prg[j] * b_strides[j] * (b_sizeof as i64);
            }
            amount += end - start;
            a_ptrs.push(a_ptr);
            b_ptrs.push(b_ptr);
            a_ptr = lhs.data.clone();
            b_ptr = rhs.data.clone();
            prgs.push(prg);
        }
        THREAD_POOL.with_borrow(|pool| {
            let iter = num_threads_each
                .into_iter()
                .zip(intervals)
                .zip(res_ptrs)
                .zip(a_ptrs)
                .zip(b_ptrs)
                .zip(prgs);

            let lhs_strides = lhs.strides().clone();
            let rhs_strides = rhs.strides().clone();
            let out_strides = res.strides().clone();
            let lhs_shape = lhs.shape().clone();
            let rhs_shape = rhs.shape().clone();
            let out_shape = res.shape().clone();
            pool.parallel_for(
                iter,
                move |(((((threads, interval), mut res_ptr), mut a_ptr), mut b_ptr), mut prg),
                      _| {
                    for _ in interval.0..interval.1 {
                        matmul(
                            a_ptr.cast::<T>(),
                            b_ptr.cast::<T>(),
                            res_ptr.cast::<T>(),
                            &lhs_strides,
                            &rhs_strides,
                            &out_strides,
                            &lhs_shape,
                            &rhs_shape,
                            &out_shape,
                            threads,
                            post_op.clone(),
                            post_op_vec.clone(),
                        );
                        res_ptr.add(res_inner_matrix_size);
                        for j in 0..iterate_shape.len() {
                            if prg[j] < iterate_shape[j] {
                                prg[j] += 1;
                                a_ptr += a_strides[j];
                                b_ptr += b_strides[j];
                                break;
                            } else {
                                prg[j] = 0;
                                a_ptr += -a_strides[j] * iterate_shape[j];
                                b_ptr += -b_strides[j] * iterate_shape[j];
                            }
                        }
                    }
                },
            );
        });
        Ok(res)
    }
}

impl Tensor {
    pub fn matmul<T>(&self, rhs: &Tensor) -> Result<Tensor, TensorError>
    where
        T: MatmulMicroKernel,
    {
        macro_rules! matmul {
            ($dtype:ty) => {
                matmul_with_out(
                    self,
                    rhs,
                    None,
                    None::<fn($dtype, usize, usize) -> $dtype>,
                    None::<
                        fn(
                            <$dtype as TypeCommon>::Vec,
                            usize,
                            usize,
                        ) -> <$dtype as TypeCommon>::Vec,
                    >,
                )
            };
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => matmul!(i8),
            hpt_types::dtype::DType::U8 => matmul!(u8),
            hpt_types::dtype::DType::F32 => matmul!(f32),
            hpt_types::dtype::DType::F16 => matmul!(f16),
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            _ => panic!("Unsupported dtype for matmul"),
        }
    }
    pub fn matmul_<T>(&self, rhs: &Tensor, out: &mut Tensor) -> Result<Tensor, TensorError>
    where
        T: MatmulMicroKernel,
    {
        macro_rules! matmul {
            ($dtype:ty) => {
                matmul_with_out(
                    self,
                    rhs,
                    Some(out.clone()),
                    None::<fn($dtype, usize, usize) -> $dtype>,
                    None::<
                        fn(
                            <$dtype as TypeCommon>::Vec,
                            usize,
                            usize,
                        ) -> <$dtype as TypeCommon>::Vec,
                    >,
                )
            };
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => matmul!(i8),
            hpt_types::dtype::DType::U8 => matmul!(u8),
            hpt_types::dtype::DType::F32 => matmul!(f32),
            hpt_types::dtype::DType::F16 => matmul!(f16),
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            _ => panic!("Unsupported dtype for matmul"),
        }
    }
    pub fn matmul_post<T, F1, F2>(
        &self,
        rhs: &Tensor,
        post_op: F1,
        post_op_vec: F2,
    ) -> Result<Tensor, TensorError>
    where
        T: CommonBounds + MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
    {
        matmul_with_out(self, rhs, None, Some(post_op), Some(post_op_vec))
    }
    pub fn matmul_post_<T, F1, F2>(
        &self,
        rhs: &Tensor,
        out: &mut Tensor,
        post_op: F1,
        post_op_vec: F2,
    ) -> Result<Tensor, TensorError>
    where
        T: CommonBounds + MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
    {
        matmul_with_out(
            self,
            rhs,
            Some(out.clone()),
            Some(post_op),
            Some(post_op_vec),
        )
    }
    pub fn addmm(&self, rhs: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
        let bias_strides = bias.strides();
        if bias.ndim() > 2 {
            panic!("bias must be a 2D tensor");
        }
        let bias_cs = bias_strides[bias_strides.len() - 1];
        let bias_rs = if bias.ndim() == 1 {
            0i64
        } else {
            bias_strides[bias_strides.len() - 2]
        };

        macro_rules! addmm {
            ($dtype:ty) => {{
                let bias_ptr = bias.ptr::<$dtype>();
                self.matmul_post::<$dtype, _, _>(
                    rhs,
                    move |inp, m, n| bias_ptr[(m as i64) * bias_rs + (n as i64)]._add(inp),
                    move |inp, m, n| {
                        let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                        (bias_ptr + offset)
                            .cast::<<$dtype as TypeCommon>::Vec>()
                            .read_unaligned()
                            ._add(inp)
                    },
                )
            }};
        }

        match self.dtype {
            hpt_types::dtype::DType::I8 => addmm!(i8),
            hpt_types::dtype::DType::U8 => addmm!(u8),
            hpt_types::dtype::DType::F32 => addmm!(f32),
            hpt_types::dtype::DType::F16 => addmm!(f16),
            hpt_types::dtype::DType::BF16 => addmm!(bf16),
            _ => panic!("Unsupported dtype for addmm"),
        }
    }
}
