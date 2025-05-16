use std::sync::Arc;

use hpt_common::{
    error::{ base::TensorError, shape::ShapeError },
    shape::shape_utils::predict_broadcast_shape,
};
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

use crate::{ Tensor, current_num_threads };

#[cfg(feature = "bf16")]
use super::matmul_mp::bf16_matmul_mp_no_block_info;
#[cfg(feature = "f16")]
use super::matmul_mp::f16_matmul_mp_no_block_info;
#[cfg(feature = "bf16")]
use super::matmul_mp_post::bf16_matmul_mp_post_no_block_info;
#[cfg(feature = "f16")]
use super::matmul_mp_post::f16_matmul_mp_post_no_block_info;
use super::{ common::matmul_prepare, microkernel_trait::MatmulMicroKernel };

use hpt_types::{ dtype::TypeCommon, traits::VecTrait, type_promote::NormalOut };

#[cfg(feature = "bf16")]
use half::bf16;
#[cfg(feature = "f16")]
use half::f16;

use hpt_types::into_scalar::Cast;

#[track_caller]
pub(crate) fn matmul_with_out<T, F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    out: Option<Tensor>,
    mut threads: usize,
    prepacked_rhs: Option<Arc<Vec<hpt_matmul::NewPrePackedRhs>>>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>
)
    -> std::result::Result<Tensor, TensorError>
    where
        T: CommonBounds + hpt_matmul::MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    threads = threads.min(num_cpus::get_physical());
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        let c = matmul_prepare(&lhs, &rhs, out)?;
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let lhs_ptr = lhs.data.cast::<T>();
        let rhs_ptr = rhs.data.cast::<T>();
        let res_ptr = c.data.cast::<T>();
        hpt_matmul::matmul::<T>(
            (lhs_ptr.ptr as *const T, lhs_ptr.len()),
            (rhs_ptr.ptr as *const T, rhs_ptr.len()),
            (res_ptr.ptr as *mut T, res_ptr.len()),
            n,
            m,
            k,
            [lhs.strides()[lhs.strides().len() - 2], lhs.strides()[lhs.strides().len() - 1]],
            [rhs.strides()[rhs.strides().len() - 2], rhs.strides()[rhs.strides().len() - 1]],
            [c.strides()[c.strides().len() - 2], c.strides()[c.strides().len() - 1]],
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec
        );
        Ok(c)
    } else {
        fn extract_batch_and_matrix_dims(
            tensor: &Tensor
        ) -> Result<(Vec<i64>, i64, i64), TensorError> {
            let shape = tensor.shape();
            let ndim = shape.len();
            if ndim < 2 {
                panic!("Tensor must have at least 2 dimensions, got {}", ndim);
            }
            let matrix_dims = &shape[ndim - 2..];
            let batch_dims = &shape[..ndim - 2];

            Ok((batch_dims.to_vec(), matrix_dims[0], matrix_dims[1]))
        }
        let (batch_dims_lhs, m, k_lhs) = extract_batch_and_matrix_dims(lhs)?;
        let (batch_dims_rhs, k_rhs, n) = extract_batch_and_matrix_dims(rhs)?;
        if k_lhs != k_rhs {
            panic!("Inner dimensions mismatch: {} vs {}", k_lhs, k_rhs);
        }
        let k = k_lhs;
        let broadcast_batch_shape = predict_broadcast_shape(&batch_dims_lhs, &batch_dims_rhs)?;
        let mut new_lhs_shape = broadcast_batch_shape.to_vec();
        new_lhs_shape.push(m);
        new_lhs_shape.push(k);
        let mut new_rhs_shape = broadcast_batch_shape.to_vec();
        new_rhs_shape.push(k);
        new_rhs_shape.push(n);
        let broacasted_lhs = lhs.broadcast_to(&new_lhs_shape)?;
        let broacasted_rhs = rhs.broadcast_to(&new_rhs_shape)?;
        let batch_size = broadcast_batch_shape.iter().product::<i64>() as usize;
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        let result = match out {
            Some(out) => {
                ShapeError::check_inplace_out_layout_valid(&output_shape, &out.layout)?;
                out
            }
            None => Tensor::empty(&output_shape, lhs.dtype, lhs.device.clone())?,
        };

        let lhs_strides: &[i64] = &broacasted_lhs.strides()[broacasted_lhs.strides().len() - 2..];
        let rhs_strides: &[i64] = &broacasted_rhs.strides()[broacasted_rhs.strides().len() - 2..];
        let out_strides: &[i64] = &result.strides()[result.strides().len() - 2..];

        let res_gp = result.map_global_idx.as_ref();
        let lhs_gp = broacasted_lhs.map_global_idx.as_ref();
        let rhs_gp = broacasted_rhs.map_global_idx.as_ref();
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        if let Some(_) = prepacked_rhs {
            panic!("prepacked_rhs is not supported for batch matmul");
        } else {
            (0..batch_size).into_par_iter().for_each(|i| {
                let i = i as i64;
                let lhs_offset = lhs_gp(i * m * k);
                let rhs_offset = rhs_gp(i * k * n);
                let out_offset = res_gp(i * m * n);
                let lhs_ptr = lhs.data.cast::<T>() + lhs_offset;
                let rhs_ptr = rhs.data.cast::<T>() + rhs_offset;
                let res_ptr = result.data.cast::<T>() + out_offset;
                hpt_matmul::matmul::<T>(
                    (lhs_ptr.ptr as *const T, lhs_ptr.len()),
                    (rhs_ptr.ptr as *const T, rhs_ptr.len()),
                    (res_ptr.ptr as *mut T, res_ptr.len()),
                    n as usize,
                    m as usize,
                    k as usize,
                    [lhs_strides[lhs_strides.len() - 2], lhs_strides[lhs_strides.len() - 1]],
                    [rhs_strides[rhs_strides.len() - 2], rhs_strides[rhs_strides.len() - 1]],
                    [out_strides[out_strides.len() - 2], out_strides[out_strides.len() - 1]],
                    1,
                    None,
                    post_op,
                    post_op_vec
                );
            });
        }

        Ok(result)
    }
}

#[track_caller]
pub(crate) fn addmm_with_out<T, F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: &Tensor,
    out: Option<Tensor>,
    mut threads: usize,
    prepacked_rhs: Option<Arc<Vec<hpt_matmul::NewPrePackedRhs>>>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>
)
    -> std::result::Result<Tensor, TensorError>
    where
        T: CommonBounds + hpt_matmul::MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    threads = threads.min(num_cpus::get_physical());
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    let bias_strides = bias.strides();
    let bias_strides = if bias_strides.len() == 1 {
        [0, bias_strides[0] as i64]
    } else if bias_strides.len() == 2 {
        if bias.shape()[0] == 1 {
            [0, bias_strides[1] as i64]
        } else if bias.shape()[0] == m as i64 {
            [bias_strides[0] as i64, bias_strides[1] as i64]
        } else {
            panic!("for 2D bias, first dim should be 1 or equal to m({}), but got shape({})", m, bias.shape());
        }
    } else {
        panic!("bias must be 1D or 2D tensor, got {}", bias_strides.len());
    };
    let bias_size = bias.size() as i64;
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        let c = matmul_prepare(&lhs, &rhs, out)?;
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let lhs_ptr = lhs.data.cast::<T>();
        let rhs_ptr = rhs.data.cast::<T>();
        let res_ptr = c.data.cast::<T>();
        hpt_matmul::addmm::<T>(
            (lhs_ptr.ptr as *const T, lhs_ptr.len()),
            (rhs_ptr.ptr as *const T, rhs_ptr.len()),
            (res_ptr.ptr as *mut T, res_ptr.len()),
            (bias.data.cast::<T>().ptr as *const T, bias_size),
            n,
            m,
            k,
            [lhs.strides()[lhs.strides().len() - 2], lhs.strides()[lhs.strides().len() - 1]],
            [rhs.strides()[rhs.strides().len() - 2], rhs.strides()[rhs.strides().len() - 1]],
            [c.strides()[c.strides().len() - 2], c.strides()[c.strides().len() - 1]],
            bias_strides,
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec
        );
        Ok(c)
    } else {
        fn extract_batch_and_matrix_dims(
            tensor: &Tensor
        ) -> Result<(Vec<i64>, i64, i64), TensorError> {
            let shape = tensor.shape();
            let ndim = shape.len();
            if ndim < 2 {
                panic!("Tensor must have at least 2 dimensions, got {}", ndim);
            }
            let matrix_dims = &shape[ndim - 2..];
            let batch_dims = &shape[..ndim - 2];

            Ok((batch_dims.to_vec(), matrix_dims[0], matrix_dims[1]))
        }
        let (batch_dims_lhs, m, k_lhs) = extract_batch_and_matrix_dims(lhs)?;
        let (batch_dims_rhs, k_rhs, n) = extract_batch_and_matrix_dims(rhs)?;
        if k_lhs != k_rhs {
            panic!("Inner dimensions mismatch: {} vs {}", k_lhs, k_rhs);
        }
        let k = k_lhs;
        let broadcast_batch_shape = predict_broadcast_shape(&batch_dims_lhs, &batch_dims_rhs)?;
        let mut new_lhs_shape = broadcast_batch_shape.to_vec();
        new_lhs_shape.push(m);
        new_lhs_shape.push(k);
        let mut new_rhs_shape = broadcast_batch_shape.to_vec();
        new_rhs_shape.push(k);
        new_rhs_shape.push(n);
        let broacasted_lhs = lhs.broadcast_to(&new_lhs_shape)?;
        let broacasted_rhs = rhs.broadcast_to(&new_rhs_shape)?;
        let batch_size = broadcast_batch_shape.iter().product::<i64>() as usize;
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        let result = match out {
            Some(out) => {
                ShapeError::check_inplace_out_layout_valid(&output_shape, &out.layout)?;
                out
            }
            None => Tensor::empty(&output_shape, lhs.dtype, lhs.device.clone())?,
        };

        let lhs_strides: &[i64] = &broacasted_lhs.strides()[broacasted_lhs.strides().len() - 2..];
        let rhs_strides: &[i64] = &broacasted_rhs.strides()[broacasted_rhs.strides().len() - 2..];
        let out_strides: &[i64] = &result.strides()[result.strides().len() - 2..];

        let res_gp = result.map_global_idx.as_ref();
        let lhs_gp = broacasted_lhs.map_global_idx.as_ref();
        let rhs_gp = broacasted_rhs.map_global_idx.as_ref();
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        if let Some(_) = prepacked_rhs {
            panic!("prepacked_rhs is not supported for batch matmul");
        } else {
            (0..batch_size).into_par_iter().for_each(|i| {
                let i = i as i64;
                let lhs_offset = lhs_gp(i * m * k);
                let rhs_offset = rhs_gp(i * k * n);
                let out_offset = res_gp(i * m * n);
                let lhs_ptr = lhs.data.cast::<T>() + lhs_offset;
                let rhs_ptr = rhs.data.cast::<T>() + rhs_offset;
                let res_ptr = result.data.cast::<T>() + out_offset;
                hpt_matmul::addmm::<T>(
                    (lhs_ptr.ptr as *const T, lhs_ptr.len()),
                    (rhs_ptr.ptr as *const T, rhs_ptr.len()),
                    (res_ptr.ptr as *mut T, res_ptr.len()),
                    (bias.data.cast::<T>().ptr as *const T, bias_size),
                    n as usize,
                    m as usize,
                    k as usize,
                    [lhs_strides[lhs_strides.len() - 2], lhs_strides[lhs_strides.len() - 1]],
                    [rhs_strides[rhs_strides.len() - 2], rhs_strides[rhs_strides.len() - 1]],
                    [out_strides[out_strides.len() - 2], out_strides[out_strides.len() - 1]],
                    bias_strides,
                    threads,
                    prepacked_rhs.clone(),
                    post_op,
                    post_op_vec
                );
            });
        }

        Ok(result)
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
                {
                    matmul_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                        self,
                        rhs,
                        None,
                        current_num_threads(),
                        None,
                        None,
                        None
                    )
                }
            };
        }
        match self.dtype {
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub fn gemm(
        &self,
        rhs: &Tensor,
        bias: Option<&Tensor>,
        alpha: f64,
        beta: f64
    ) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
        {
                type T = $dtype;
                type InVec = <T as TypeCommon>::Vec;
                if let Some(bias) = bias {
                    let bias_ptr = bias.ptr::<T>();
                    let bias_strides = bias.strides();
                    let bias_cs = bias_strides[bias_strides.len() - 1];
                    let bias_rs = if bias.ndim() == 1 {
                        0i64
                    } else {
                        bias_strides[bias_strides.len() - 2]
                    };
                    let alpha_casted: T = alpha.cast();
                    let beta_casted: T = beta.cast();
                    let alpha_vec = InVec::splat(alpha_casted);
                    let beta_vec = InVec::splat(beta_casted);
                    match (alpha == 1.0, beta == 0.0) {
                        (true, true) => matmul_with_out(
                            self,
                            rhs,
                            None,
                            current_num_threads(),
                            None,
                            None::<fn(T, usize, usize) -> T>,
                            None::<fn(InVec, usize, usize) -> InVec>,
                        ),
                        (true, false) => {
                            matmul_with_out(
                                self,
                                rhs,
                                None,
                                current_num_threads(),
                                None,
                                Some(move |inp: T, m, n| {
                                    inp._add(beta_casted._mul(
                                        bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs],
                                    ))
                                }),
                                Some(move |inp: InVec, m, n| {
                                    let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                                    (bias_ptr + offset).cast::<InVec>().read_unaligned() * beta_vec
                                        + inp
                                }),
                            )
                        }
                        (false, true) => matmul_with_out(
                            self,
                            rhs,
                            None,
                            current_num_threads(),
                            None,
                            Some(move |inp: T, _, _| inp._mul(alpha_casted)),
                            Some(move |inp: InVec, _, _| inp * alpha_vec),
                        ),
                        (false, false) => {
                            matmul_with_out(
                                self,
                                rhs,
                                None,
                                current_num_threads(),
                                None,
                                Some(move |inp: T, m, n| {
                                    inp._mul(alpha_casted)._add(beta_casted._mul(
                                        bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs],
                                    ))
                                }),
                                Some(move |inp: InVec, m, n| {
                                    let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                                    (bias_ptr + offset).cast::<InVec>().read_unaligned() * beta_vec
                                        + inp * alpha_vec
                                }),
                            )
                        }
                    }
                } else {
                    if alpha == 1.0 {
                        matmul_with_out(
                            self,
                            rhs,
                            None,
                            current_num_threads(),
                            None,
                            None::<fn(T, usize, usize) -> T>,
                            None::<fn(InVec, usize, usize) -> InVec>,
                        )
                    } else {
                        let alpha_casted: T = alpha.cast();
                        let alpha_vec = InVec::splat(alpha_casted);
                        matmul_with_out(
                            self,
                            rhs,
                            None,
                            current_num_threads(),
                            None,
                            Some(move |inp: T, _, _| inp._mul(alpha_casted)),
                            Some(move |inp: InVec, _, _| inp * alpha_vec),
                        )
                    }
                }
        }
            };
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub fn matmul_(&self, rhs: &Tensor, out: &mut Tensor) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
                matmul_with_out(
                    self,
                    rhs,
                    Some(out.clone()),
                    current_num_threads(),
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
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub fn addmm(&self, rhs: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
                {
                    addmm_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                        self,
                        rhs,
                        bias,
                        None,
                        current_num_threads(),
                        None,
                        None,
                        None
                    )
                }
            };
        }
        match self.dtype {
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub fn addmm_<T>(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: &mut Tensor
    ) -> Result<Tensor, TensorError>
        where T: MatmulMicroKernel
    {
        macro_rules! matmul {
            ($dtype:ty) => {
                addmm_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                    self,
                    rhs,
                    bias,
                    Some(out.clone()),
                    current_num_threads(),
                    None,
                    None,
                    None
                )
            };
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub(crate) fn addmm_prepacked_(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: &mut Tensor,
        num_threads: usize,
        prepacked_rhs: Option<Arc<Vec<hpt_matmul::NewPrePackedRhs>>>
    ) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
                addmm_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                    self,
                    rhs,
                    bias,
                    Some(out.clone()),
                    num_threads,
                    prepacked_rhs,
                    None,
                    None
                )
            };
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => matmul!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => matmul!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => matmul!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => matmul!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => matmul!(bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => matmul!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => matmul!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => matmul!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => matmul!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => matmul!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => matmul!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => matmul!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => matmul!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }
}
