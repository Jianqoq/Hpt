use std::borrow::{Borrow, BorrowMut};

use crate::backends::cpu::kernels::matmul::matmul::{matmul, matmul_template_no_block_info};
use crate::backends::cpu::kernels::matmul::matmul_mixed_precision::{
    bf16_matmul_mixed_precision_template_no_block_info,
    f16_matmul_mixed_precision_template_no_block_info,
};
use crate::backends::cpu::kernels::matmul::matmul_mp_post::{
    bf16_matmul_mp_post_template_no_block_info, f16_matmul_mp_post_template_no_block_info,
};
use crate::backends::cpu::kernels::matmul::matmul_post::{
    matmul_post, matmul_post_template_no_block_info,
};
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::tensor_base::_Tensor;
use crate::utils::get_num_threads;
use crate::CUSTOM_THREAD_POOL;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::shape::shape_utils::{compare_and_pad_shapes, mt_intervals};
use hpt_common::strides::strides_utils::preprocess_strides;
use hpt_traits::ops::binary::{Matmul, MatmulPost};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};

#[track_caller]
pub(crate) fn matmul_with_out<T, O, A2, const DEVICE: usize, F1, F2>(
    lhs: &_Tensor<T, Cpu, DEVICE, A2>,
    rhs: &_Tensor<T, Cpu, DEVICE, A2>,
    out: Option<O>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) -> std::result::Result<_Tensor<T, Cpu, DEVICE, A2>, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    O: BorrowMut<_Tensor<T, Cpu, DEVICE, A2>>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        match (post_op, post_op_vec) {
            (None, None) => {
                #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
                {
                    if T::STR == "bf16" {
                        use crate::backends::cpu::kernels::matmul::matmul_mixed_precision::bf16_matmul;
                        let res = bf16_matmul(
                            &lhs.static_cast::<half::bf16>()?,
                            &rhs.static_cast::<half::bf16>()?,
                            out.map(|mut x| {
                                x.borrow_mut()
                                    .clone()
                                    .static_cast::<half::bf16>()
                                    .expect("static_cast bf16 failed")
                            }),
                            rayon::current_num_threads(),
                        )?;
                        Ok(res.static_cast::<T>()?)
                    } else {
                        matmul(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            rayon::current_num_threads(),
                        )
                    }
                }
                #[cfg(not(all(
                    target_feature = "neon",
                    target_arch = "aarch64",
                    target_feature = "fp16"
                )))]
                {
                    match T::STR {
                        "f16" => {
                            use crate::backends::cpu::kernels::matmul::matmul_mixed_precision::f16_matmul;
                            let res = f16_matmul(
                                &lhs.static_cast::<half::f16>()?,
                                &rhs.static_cast::<half::f16>()?,
                                out.map(|mut x| {
                                    x.borrow_mut()
                                        .clone()
                                        .static_cast::<half::f16>()
                                        .expect("static_cast f16 failed")
                                }),
                                get_num_threads(),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        "bf16" => {
                            use crate::backends::cpu::kernels::matmul::matmul_mixed_precision::bf16_matmul;
                            let res = bf16_matmul(
                                &lhs.static_cast::<half::bf16>()?,
                                &rhs.static_cast::<half::bf16>()?,
                                out.map(|mut x| {
                                    x.borrow_mut()
                                        .clone()
                                        .static_cast::<half::bf16>()
                                        .expect("static_cast bf16 failed")
                                }),
                                get_num_threads(),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        _ => matmul(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            get_num_threads(),
                        ),
                    }
                }
            }
            (Some(post_op), Some(post_op_vec)) => {
                #[cfg(all(
                    target_feature = "neon",
                    target_arch = "aarch64",
                    target_feature = "fp16"
                ))]
                {
                    if T::STR == "bf16" {
                        use crate::backends::cpu::kernels::matmul::matmul_mp_post::bf16_matmul_post;
                        let res = bf16_matmul_post(
                            &lhs,
                            &rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            post_op,
                            post_op_vec,
                            rayon::current_num_threads(),
                        )?;
                        Ok(res.static_cast::<T>()?)
                    } else {
                        matmul_post(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            post_op,
                            post_op_vec,
                            rayon::current_num_threads(),
                        )
                    }
                }
                #[cfg(not(all(
                    target_feature = "neon",
                    target_arch = "aarch64",
                    target_feature = "fp16"
                )))]
                {
                    match T::STR {
                        "f16" => {
                            use crate::backends::cpu::kernels::matmul::matmul_mp_post::f16_matmul_post;
                            let res = f16_matmul_post(
                                &lhs,
                                &rhs,
                                out.map(|mut x| x.borrow_mut().clone()),
                                post_op,
                                post_op_vec,
                                get_num_threads(),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        "bf16" => {
                            use crate::backends::cpu::kernels::matmul::matmul_mp_post::bf16_matmul_post;
                            let res = bf16_matmul_post(
                                &lhs,
                                &rhs,
                                out.map(|mut x| x.borrow_mut().clone()),
                                post_op,
                                post_op_vec,
                                get_num_threads(),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        _ => matmul_post(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            post_op,
                            post_op_vec,
                            get_num_threads(),
                        ),
                    }
                }
            }
            _ => panic!("post_op and post_op_vec must be both Some or both None"),
        }
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
        let res = if let Some(mut out) = out {
            let out: _Tensor<T, Cpu, DEVICE, A2> = out.borrow_mut().clone();
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            out
        } else {
            _Tensor::<T, Cpu, DEVICE, A2>::empty(res_shape)?
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
        let rayon_num_threads = CUSTOM_THREAD_POOL.with_borrow(|pool| pool.num_threads());
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
            res_ptr.add((end - start) * res_inner_matrix_size);
            let mut prg = vec![0i64; iterate_shape.len()];
            let mut amount_cpy = amount as i64;
            for j in (0..=iterate_shape.len() - 1).rev() {
                prg[j] = amount_cpy % (iterate_shape[j] + 1);
                amount_cpy /= iterate_shape[j] + 1;
                a_ptr.offset(prg[j] * a_strides[j]);
                b_ptr.offset(prg[j] * b_strides[j]);
            }
            amount += end - start;
            a_ptrs.push(a_ptr);
            b_ptrs.push(b_ptr);
            a_ptr = lhs.data.clone();
            b_ptr = rhs.data.clone();
            prgs.push(prg);
        }
        let lhs_cs = lhs.strides()[lhs.strides().len() - 1];
        let lhs_rs = lhs.strides()[lhs.strides().len() - 2];
        let dst_cs = res.strides()[res.strides().len() - 2];
        let rhs_cs = rhs.strides()[rhs.strides().len() - 1];
        let rhs_rs = rhs.strides()[rhs.strides().len() - 2];
        let m = a_shape[a_shape.len() - 2] as usize;
        let n = b_shape[b_shape.len() - 1] as usize;
        let k = b_shape[b_shape.len() - 2] as usize;

        CUSTOM_THREAD_POOL.with_borrow(|pool| {
            let iter = num_threads_each
                .into_iter()
                .zip(intervals)
                .zip(res_ptrs)
                .zip(a_ptrs)
                .zip(b_ptrs)
                .zip(prgs);
            pool.parallel_for(
                iter,
                move |(
                    ((((threads, (start, end)), mut res_ptr), mut a_ptr), mut b_ptr),
                    mut prg,
                ),
                      _| {
                    for _ in start..end {
                        match T::STR {
                            "f16" => {
                                if let (Some(post_op), Some(post_op_vec)) =
                                    (post_op.clone(), post_op_vec.clone())
                                {
                                    f16_matmul_mp_post_template_no_block_info::<T, f32, _, _>(
                                        a_ptr,
                                        b_ptr,
                                        res_ptr,
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
                                    );
                                } else {
                                    f16_matmul_mixed_precision_template_no_block_info::<T, f32>(
                                        a_ptr, b_ptr, res_ptr, m, n, k, lhs_rs, rhs_rs, dst_cs,
                                        lhs_cs, rhs_cs, threads,
                                    );
                                }
                            }
                            "bf16" => {
                                if let (Some(post_op), Some(post_op_vec)) =
                                    (post_op.clone(), post_op_vec.clone())
                                {
                                    bf16_matmul_mp_post_template_no_block_info::<T, f32, _, _>(
                                        a_ptr,
                                        b_ptr,
                                        res_ptr,
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
                                    );
                                } else {
                                    bf16_matmul_mixed_precision_template_no_block_info::<T, f32>(
                                        a_ptr, b_ptr, res_ptr, m, n, k, lhs_rs, rhs_rs, dst_cs,
                                        lhs_cs, rhs_cs, threads,
                                    );
                                }
                            }
                            _ => {
                                if let (Some(post_op), Some(post_op_vec)) =
                                    (post_op.clone(), post_op_vec.clone())
                                {
                                    matmul_post_template_no_block_info(
                                        a_ptr.clone(),
                                        b_ptr.clone(),
                                        res_ptr.clone(),
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
                                    );
                                } else {
                                    matmul_template_no_block_info(
                                        a_ptr.clone(),
                                        b_ptr.clone(),
                                        res_ptr.clone(),
                                        m,
                                        n,
                                        k,
                                        lhs_rs,
                                        rhs_rs,
                                        dst_cs,
                                        lhs_cs,
                                        rhs_cs,
                                        threads,
                                    );
                                }
                            }
                        }
                        res_ptr.add(res_inner_matrix_size);
                        for j in 0..iterate_shape.len() {
                            if prg[j] < iterate_shape[j] {
                                prg[j] += 1;
                                a_ptr.offset(a_strides[j]);
                                b_ptr.offset(b_strides[j]);
                                break;
                            } else {
                                prg[j] = 0;
                                a_ptr.offset(-a_strides[j] * iterate_shape[j]);
                                b_ptr.offset(-b_strides[j] * iterate_shape[j]);
                            }
                        }
                    }
                },
            );
        });
        Ok(res)
    }
}

impl<T, A2, const DEVICE: usize> Matmul<_Tensor<T, Cpu, DEVICE, A2>> for _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + MatmulMicroKernel,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A2>;

    type OutputMeta = T;

    type InplaceOutput = _Tensor<T, Cpu, DEVICE, A2>;

    fn matmul(&self, rhs: _Tensor<T, Cpu, DEVICE, A2>) -> Result<Self::Output, TensorError> {
        matmul_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            None::<fn(T, usize, usize) -> T>,
            None::<fn(T::Vec, usize, usize) -> T::Vec>,
        )
    }
    fn matmul_<U>(
        &self,
        rhs: _Tensor<T, Cpu, DEVICE, A2>,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(
            self,
            &rhs,
            Some(out),
            None::<fn(T, usize, usize) -> T>,
            None::<fn(T::Vec, usize, usize) -> T::Vec>,
        )
    }

    fn addmm(
        &self,
        rhs: _Tensor<T, Cpu, DEVICE, A2>,
        bias: _Tensor<T, Cpu, DEVICE, A2>,
    ) -> std::result::Result<Self::Output, TensorError> {
        todo!()
    }
}

impl<T, A2, const DEVICE: usize> Matmul<&_Tensor<T, Cpu, DEVICE, A2>>
    for _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + MatmulMicroKernel,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A2>;

    type OutputMeta = T;

    type InplaceOutput = _Tensor<T, Cpu, DEVICE, A2>;

    fn matmul(&self, rhs: &_Tensor<T, Cpu, DEVICE, A2>) -> Result<Self::Output, TensorError> {
        matmul_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            None::<fn(T, usize, usize) -> T>,
            None::<fn(T::Vec, usize, usize) -> T::Vec>,
        )
    }

    fn matmul_<U>(
        &self,
        rhs: &_Tensor<T, Cpu, DEVICE, A2>,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(
            self,
            rhs,
            Some(out),
            None::<fn(T, usize, usize) -> T>,
            None::<fn(T::Vec, usize, usize) -> T::Vec>,
        )
    }

    fn addmm(
        &self,
        rhs: &_Tensor<T, Cpu, DEVICE, A2>,
        bias: &_Tensor<T, Cpu, DEVICE, A2>,
    ) -> std::result::Result<Self::Output, TensorError> {
        todo!()
    }
}

impl<T, A2, const DEVICE: usize> MatmulPost<_Tensor<T, Cpu, DEVICE, A2>>
    for _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + MatmulMicroKernel,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A2>;

    type OutputMeta = T;

    type InplaceOutput = _Tensor<T, Cpu, DEVICE, A2>;

    fn matmul_post(
        &self,
        rhs: _Tensor<T, Cpu, DEVICE, A2>,
        post_op: fn(T, usize, usize) -> T,
        post_op_vec: fn(T::Vec, usize, usize) -> T::Vec,
    ) -> std::result::Result<Self::Output, TensorError> {
        matmul_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            Some(post_op),
            Some(post_op_vec),
        )
    }

    fn matmul_post_<U>(
        &self,
        rhs: _Tensor<T, Cpu, DEVICE, A2>,
        post_op: fn(T, usize, usize) -> T,
        post_op_vec: fn(T::Vec, usize, usize) -> T::Vec,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, Some(out), Some(post_op), Some(post_op_vec))
    }

    fn addmm_post<F, F2>(
        &self,
        rhs: _Tensor<T, Cpu, DEVICE, A2>,
        bias: _Tensor<T, Cpu, DEVICE, A2>,
        post_op: F,
        post_op_vec: F2,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        F: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
    {
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
        let ptr = bias.ptr();
        if bias_cs == 1 {
            matmul_with_out(
                self,
                &rhs,
                None::<Self::Output>,
                Some(move |x: T, m: usize, n: usize| {
                    let val = ptr[m as i64 * bias_rs + n as i64 * bias_cs];
                    post_op(x._add(val), m, n)
                }),
                Some(move |x: T::Vec, mm: usize, nn: usize| {
                    use hpt_types::type_promote::NormalOut;
                    let offset = mm as i64 * bias_rs + nn as i64 * bias_cs;
                    let vec = ptr
                        .cast::<T>()
                        .offset(offset)
                        .cast::<T::Vec>()
                        .read_unaligned();
                    post_op_vec(x._add(vec), mm, nn)
                }),
            )
        } else {
            matmul_with_out(
                self,
                &rhs,
                None::<Self::Output>,
                Some(move |x: T, m: usize, n: usize| {
                    let val = ptr[m as i64 * bias_rs + n as i64 * bias_cs];
                    post_op(x._add(val), m, n)
                }),
                Some(move |x: T::Vec, mm: usize, nn: usize| {
                    use hpt_types::traits::VecTrait;
                    use hpt_types::type_promote::NormalOut;
                    let offset = mm as i64 * bias_rs + nn as i64 * bias_cs;
                    let ptr = ptr.cast::<T>().offset(offset);
                    let mut vec = T::Vec::splat(T::ZERO);
                    for i in 0..T::Vec::SIZE {
                        vec[i] = ptr[i];
                    }
                    post_op_vec(x._add(vec), mm, nn)
                }),
            )
        }
    }
}

impl<T, A2, const DEVICE: usize> MatmulPost<&_Tensor<T, Cpu, DEVICE, A2>>
    for _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + MatmulMicroKernel,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A2>;

    type OutputMeta = T;

    type InplaceOutput = _Tensor<T, Cpu, DEVICE, A2>;

    fn matmul_post(
        &self,
        rhs: &_Tensor<T, Cpu, DEVICE, A2>,
        post_op: fn(T, usize, usize) -> T,
        post_op_vec: fn(T::Vec, usize, usize) -> T::Vec,
    ) -> std::result::Result<Self::Output, TensorError> {
        matmul_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            Some(post_op),
            Some(post_op_vec),
        )
    }

    fn matmul_post_<U>(
        &self,
        rhs: &_Tensor<T, Cpu, DEVICE, A2>,
        post_op: fn(T, usize, usize) -> T,
        post_op_vec: fn(T::Vec, usize, usize) -> T::Vec,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, Some(out), Some(post_op), Some(post_op_vec))
    }

    fn addmm_post<F, F2>(
        &self,
        rhs: &_Tensor<T, Cpu, DEVICE, A2>,
        bias: &_Tensor<T, Cpu, DEVICE, A2>,
        post_op: F,
        post_op_vec: F2,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        F: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
    {
        self.addmm_post(rhs.clone(), bias.clone(), post_op, post_op_vec)
    }
}
