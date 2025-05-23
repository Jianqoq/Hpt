use std::borrow::{Borrow, BorrowMut};

use crate::backends::cpu::kernels::matmul::matmul::{matmul, matmul_template_no_block_info};
use crate::backends::cpu::kernels::matmul::matmul_mixed_precision::matmul_mixed_precision_template_no_block_info;
use crate::backends::cpu::kernels::matmul::matmul_mp_post::matmul_mp_post_template_no_block_info;
use crate::backends::cpu::kernels::matmul::matmul_post::{
    matmul_post, matmul_post_template_no_block_info,
};
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::tensor_base::_Tensor;
use crate::{RAYON_NUM_THREADS, THREAD_POOL};
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
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::VecTrait;

#[track_caller]
pub(crate) fn matmul_with_out<T, O, A2, const DEVICE: usize>(
    lhs: &_Tensor<T, Cpu, DEVICE, A2>,
    rhs: &_Tensor<T, Cpu, DEVICE, A2>,
    out: Option<O>,
    post_op: Option<fn(T) -> T>,
    post_op_vec: Option<fn(T::Vec) -> T::Vec>,
) -> std::result::Result<_Tensor<T, Cpu, DEVICE, A2>, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    O: BorrowMut<_Tensor<T, Cpu, DEVICE, A2>>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        match (post_op, post_op_vec) {
            (None, None) => {
                #[cfg(all(
                    target_feature = "neon",
                    target_arch = "aarch64",
                    target_feature = "fp16"
                ))]
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
                                RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
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
                                RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        _ => matmul(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
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
                        let post_op = unsafe { std::mem::transmute(post_op) };
                        let post_op_vec = unsafe { std::mem::transmute(post_op_vec) };
                        let res = bf16_matmul_post(
                            &lhs.static_cast::<half::bf16>()?,
                            &rhs.static_cast::<half::bf16>()?,
                            out.map(|mut x| {
                                x.borrow_mut()
                                    .clone()
                                    .static_cast::<half::bf16>()
                                    .expect("static_cast bf16 failed")
                            }),
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
                            let post_op = unsafe { std::mem::transmute(post_op) };
                            let post_op_vec = unsafe { std::mem::transmute(post_op_vec) };
                            let res = f16_matmul_post(
                                &lhs.static_cast::<half::f16>()?,
                                &rhs.static_cast::<half::f16>()?,
                                out.map(|mut x| {
                                    x.borrow_mut()
                                        .clone()
                                        .static_cast::<half::f16>()
                                        .expect("static_cast f16 failed")
                                }),
                                post_op,
                                post_op_vec,
                                RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        "bf16" => {
                            use crate::backends::cpu::kernels::matmul::matmul_mp_post::bf16_matmul_post;
                            let post_op = unsafe { std::mem::transmute(post_op) };
                            let post_op_vec = unsafe { std::mem::transmute(post_op_vec) };
                            let res = bf16_matmul_post(
                                &lhs.static_cast::<half::bf16>()?,
                                &rhs.static_cast::<half::bf16>()?,
                                out.map(|mut x| {
                                    x.borrow_mut()
                                        .clone()
                                        .static_cast::<half::bf16>()
                                        .expect("static_cast bf16 failed")
                                }),
                                post_op,
                                post_op_vec,
                                RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
                            )?;
                            Ok(res.static_cast::<T>()?)
                        }
                        _ => matmul_post(
                            lhs,
                            rhs,
                            out.map(|mut x| x.borrow_mut().clone()),
                            post_op,
                            post_op_vec,
                            RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed),
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
        let rayon_num_threads = RAYON_NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        let num_threads = batch.min(rayon_num_threads);
        let mut num_threads_each: Vec<usize> = if batch < rayon_num_threads {
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
            let mut prg: Vec<i64> = vec![0; iterate_shape.len()];
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
        THREAD_POOL.with_borrow_mut(|pool| {
            for i in (0..num_threads).rev() {
                let threads = num_threads_each.pop().unwrap();
                let current_size = intervals[i].1 - intervals[i].0;
                let mut res_ptr = res_ptrs.pop().unwrap();
                let mut a_ptr = a_ptrs.pop().unwrap();
                let mut b_ptr = b_ptrs.pop().unwrap();
                let mut prg = prgs.pop().unwrap();
                let shape = iterate_shape.clone();
                let __a_strides = a_strides.clone();
                let __b_strides = b_strides.clone();
                type F32Vec = <f32 as TypeCommon>::Vec;
                type F16Vec = <half::f16 as TypeCommon>::Vec;
                type BF16Vec = <half::bf16 as TypeCommon>::Vec;
                pool.execute(move || {
                    for _ in 0..current_size {
                        match T::STR {
                            "f16" => {
                                if let (Some(post_op), Some(post_op_vec)) = (post_op, post_op_vec) {
                                    matmul_mp_post_template_no_block_info::<half::f16, f32>(
                                        a_ptr.clone().cast::<half::f16>(),
                                        b_ptr.clone().cast::<half::f16>(),
                                        res_ptr.clone().cast::<half::f16>(),
                                        m,
                                        n,
                                        k,
                                        lhs_rs,
                                        rhs_rs,
                                        dst_cs,
                                        lhs_cs,
                                        rhs_cs,
                                        threads,
                                        |packed_b, b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            let b_vec = b.add(i).read_unaligned();
                                            let val_f32 = b_vec.to_2_f32vec();
                                            packed_b_vec0.write(val_f32[0]);
                                            packed_b_vec1.write(val_f32[1]);
                                        },
                                        |packed_b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            packed_b_vec0.write(F32Vec::splat(0.0));
                                            packed_b_vec1.write(F32Vec::splat(0.0));
                                        },
                                        |val| val.cast(),
                                        |val| {
                                            let vec0 = unsafe { val.read() };
                                            let vec1 = unsafe { val.add(1).read() };
                                            F16Vec::from_2_f32vec([vec0, vec1])
                                        },
                                        |val| val.cast(),
                                        unsafe { std::mem::transmute(post_op) },
                                        unsafe { std::mem::transmute(post_op_vec) },
                                    );
                                } else {
                                    matmul_mixed_precision_template_no_block_info::<half::f16, f32>(
                                        a_ptr.clone().cast::<half::f16>(),
                                        b_ptr.clone().cast::<half::f16>(),
                                        res_ptr.clone().cast::<half::f16>(),
                                        m,
                                        n,
                                        k,
                                        lhs_rs,
                                        rhs_rs,
                                        dst_cs,
                                        lhs_cs,
                                        rhs_cs,
                                        threads,
                                        |packed_b, b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            let b_vec = b.add(i).read_unaligned();
                                            let val_f32 = b_vec.to_2_f32vec();
                                            packed_b_vec0.write(val_f32[0]);
                                            packed_b_vec1.write(val_f32[1]);
                                        },
                                        |packed_b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            packed_b_vec0.write(F32Vec::splat(0.0));
                                            packed_b_vec1.write(F32Vec::splat(0.0));
                                        },
                                        |val| val.cast(),
                                        |val| {
                                            let vec0 = unsafe { val.read() };
                                            let vec1 = unsafe { val.add(1).read() };
                                            F16Vec::from_2_f32vec([vec0, vec1])
                                        },
                                        |val| val.cast(),
                                    );
                                }
                            }
                            "bf16" => {
                                if let (Some(post_op), Some(post_op_vec)) = (post_op, post_op_vec) {
                                    matmul_mp_post_template_no_block_info::<half::bf16, f32>(
                                        a_ptr.clone().cast::<half::bf16>(),
                                        b_ptr.clone().cast::<half::bf16>(),
                                        res_ptr.clone().cast::<half::bf16>(),
                                        m,
                                        n,
                                        k,
                                        lhs_rs,
                                        rhs_rs,
                                        dst_cs,
                                        lhs_cs,
                                        rhs_cs,
                                        threads,
                                        |packed_b, b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            let b_vec = b.add(i).read_unaligned();
                                            let val_f32 = b_vec.to_2_f32vec();
                                            packed_b_vec0.write(val_f32[0]);
                                            packed_b_vec1.write(val_f32[1]);
                                        },
                                        |packed_b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            packed_b_vec0.write(F32Vec::splat(0.0));
                                            packed_b_vec1.write(F32Vec::splat(0.0));
                                        },
                                        |val| val.cast(),
                                        |val| {
                                            let vec0 = unsafe { val.read() };
                                            let vec1 = unsafe { val.add(1).read() };
                                            BF16Vec::from_2_f32vec([vec0, vec1])
                                        },
                                        |val| val.cast(),
                                        unsafe { std::mem::transmute(post_op) },
                                        unsafe { std::mem::transmute(post_op_vec) },
                                    );
                                } else {
                                    matmul_mixed_precision_template_no_block_info::<half::bf16, f32>(
                                        a_ptr.clone().cast::<half::bf16>(),
                                        b_ptr.clone().cast::<half::bf16>(),
                                        res_ptr.clone().cast::<half::bf16>(),
                                        m,
                                        n,
                                        k,
                                        lhs_rs,
                                        rhs_rs,
                                        dst_cs,
                                        lhs_cs,
                                        rhs_cs,
                                        threads,
                                        |packed_b, b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            let b_vec = b.add(i).read_unaligned();
                                            let val_f32 = b_vec.to_2_f32vec();
                                            packed_b_vec0.write(val_f32[0]);
                                            packed_b_vec1.write(val_f32[1]);
                                        },
                                        |packed_b, i| unsafe {
                                            let packed_b_vec0 = packed_b.add(i * 2);
                                            let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                            packed_b_vec0.write(F32Vec::splat(0.0));
                                            packed_b_vec1.write(F32Vec::splat(0.0));
                                        },
                                        |val| val.cast(),
                                        |val| {
                                            let vec0 = unsafe { val.read() };
                                            let vec1 = unsafe { val.add(1).read() };
                                            BF16Vec::from_2_f32vec([vec0, vec1])
                                        },
                                        |val| val.cast(),
                                    );
                                }
                            }
                            _ => {
                                if let (Some(post_op), Some(post_op_vec)) = (post_op, post_op_vec) {
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
                        for j in 0..shape.len() {
                            if prg[j] < shape[j] {
                                prg[j] += 1;
                                a_ptr.offset(__a_strides[j]);
                                b_ptr.offset(__b_strides[j]);
                                break;
                            } else {
                                prg[j] = 0;
                                a_ptr.offset(-__a_strides[j] * shape[j]);
                                b_ptr.offset(-__b_strides[j] * shape[j]);
                            }
                        }
                    }
                });
            }
            pool.join();
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
            None::<fn(T) -> T>,
            None::<fn(T::Vec) -> T::Vec>,
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
            None::<fn(T) -> T>,
            None::<fn(T::Vec) -> T::Vec>,
        )
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
            None::<fn(T) -> T>,
            None::<fn(T::Vec) -> T::Vec>,
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
            None::<fn(T) -> T>,
            None::<fn(T::Vec) -> T::Vec>,
        )
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
        post_op: fn(T) -> T,
        post_op_vec: fn(T::Vec) -> T::Vec,
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
        post_op: fn(T) -> T,
        post_op_vec: fn(T::Vec) -> T::Vec,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, Some(out), Some(post_op), Some(post_op_vec))
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
        post_op: fn(T) -> T,
        post_op_vec: fn(T::Vec) -> T::Vec,
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
        post_op: fn(T) -> T,
        post_op_vec: fn(T::Vec) -> T::Vec,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, Some(out), Some(post_op), Some(post_op_vec))
    }
}
