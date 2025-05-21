use std::sync::Arc;

use hpt_common::{
    error::base::TensorError,
    layout::layout::Layout,
    shape::{ shape::Shape, shape_utils::predict_broadcast_shape },
    Pointer,
};
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

use crate::{ current_num_threads, utils::index_cal::dispatch_map_global_idx, Tensor };

#[cfg(feature = "bf16")]
use super::matmul_mp::bf16_matmul_mp_no_block_info;
#[cfg(feature = "bf16")]
use super::matmul_mp_post::bf16_matmul_mp_post_no_block_info;
use super::common::{ check_out_layout, matmul_prepare };

use hpt_types::{ dtype::{ DType, TypeCommon }, traits::VecTrait, type_promote::NormalOut };

#[cfg(feature = "bf16")]
use half::bf16;
#[cfg(feature = "f16")]
use half::f16;

use hpt_types::into_scalar::Cast;

#[track_caller]
pub(crate) fn matmul_with_out<T, F1, F2>(
    lhs: Pointer<T>,
    lhs_layout: &Layout,
    rhs: Pointer<T>,
    rhs_layout: &Layout,
    out: Pointer<T>,
    out_layout: &Layout,
    mut threads: usize,
    prepacked_rhs: Option<Arc<hpt_matmul::PrePackedRhs>>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>
)
    -> std::result::Result<(), TensorError>
    where
        T: CommonBounds + hpt_matmul::MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    threads = threads.min(num_cpus::get_physical());
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
        check_out_layout(lhs_layout, rhs_layout, out_layout)?;
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let lhs_ptr = lhs;
        let rhs_ptr = rhs;
        let res_ptr = out;
        hpt_matmul::matmul::<T>(
            (lhs_ptr.ptr as *const T, lhs_ptr.len()),
            (rhs_ptr.ptr as *const T, rhs_ptr.len()),
            (res_ptr.ptr as *mut T, res_ptr.len()),
            n,
            m,
            k,
            [
                lhs_layout.strides()[lhs_layout.ndim() - 2],
                lhs_layout.strides()[lhs_layout.ndim() - 1],
            ],
            [
                rhs_layout.strides()[rhs_layout.ndim() - 2],
                rhs_layout.strides()[rhs_layout.ndim() - 1],
            ],
            [
                out_layout.strides()[out_layout.ndim() - 2],
                out_layout.strides()[out_layout.ndim() - 1],
            ],
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec
        );
        Ok(())
    } else {
        fn extract_batch_and_matrix_dims(
            layout: &Layout
        ) -> Result<(Vec<i64>, i64, i64), TensorError> {
            let shape = layout.shape();
            let ndim = shape.len();
            if ndim < 2 {
                panic!("Tensor must have at least 2 dimensions, got {}", ndim);
            }
            let matrix_dims = &shape[ndim - 2..];
            let batch_dims = &shape[..ndim - 2];

            Ok((batch_dims.to_vec(), matrix_dims[0], matrix_dims[1]))
        }
        let (batch_dims_lhs, m, k_lhs) = extract_batch_and_matrix_dims(lhs_layout)?;
        let (batch_dims_rhs, k_rhs, n) = extract_batch_and_matrix_dims(rhs_layout)?;
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
        let broacasted_lhs = lhs_layout.broadcast_to(&Shape::from(&new_lhs_shape))?;
        let broacasted_rhs = rhs_layout.broadcast_to(&Shape::from(&new_rhs_shape))?;
        let batch_size = broadcast_batch_shape.iter().product::<i64>() as usize;
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        check_out_layout(lhs_layout, rhs_layout, out_layout)?;

        let lhs_strides: &[i64] = &broacasted_lhs.strides()[broacasted_lhs.strides().len() - 2..];
        let rhs_strides: &[i64] = &broacasted_rhs.strides()[broacasted_rhs.strides().len() - 2..];
        let out_strides: &[i64] = &out_layout.strides()[out_layout.ndim() - 2..];

        let res_gp = dispatch_map_global_idx(out_layout);
        let lhs_gp = dispatch_map_global_idx(&broacasted_lhs);
        let rhs_gp = dispatch_map_global_idx(&broacasted_rhs);
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
                let lhs_ptr = lhs + lhs_offset;
                let rhs_ptr = rhs + rhs_offset;
                let res_ptr = out + out_offset;
                hpt_matmul::matmul::<T>(
                    (lhs_ptr.cast::<T>().ptr as *const T, lhs_ptr.len()),
                    (rhs_ptr.cast::<T>().ptr as *const T, rhs_ptr.len()),
                    (res_ptr.cast::<T>().ptr as *mut T, res_ptr.len()),
                    n as usize,
                    m as usize,
                    k as usize,
                    [lhs_strides[lhs_strides.len() - 2], lhs_strides[lhs_strides.len() - 1]],
                    [rhs_strides[rhs_strides.len() - 2], rhs_strides[rhs_strides.len() - 1]],
                    [out_strides[out_strides.len() - 2], out_strides[out_strides.len() - 1]],
                    threads,
                    prepacked_rhs.clone(),
                    post_op,
                    post_op_vec
                );
            });
        }

        Ok(())
    }
}

#[track_caller]
pub(crate) fn addmm_with_out<T, F1, F2>(
    lhs: Pointer<T>,
    lhs_layout: &Layout,
    rhs: Pointer<T>,
    rhs_layout: &Layout,
    bias: Pointer<T>,
    bias_layout: &Layout,
    out: Pointer<T>,
    out_layout: &Layout,
    mut threads: usize,
    prepacked_rhs: Option<Arc<hpt_matmul::PrePackedRhs>>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>
)
    -> std::result::Result<(), TensorError>
    where
        T: CommonBounds + hpt_matmul::MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    threads = threads.min(num_cpus::get_physical());
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let bias_shape = bias_layout.shape();
    let bias_strides = bias_layout.strides();
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    let bias_strides = if bias_strides.len() == 1 {
        [0, bias_strides[0] as i64]
    } else if bias_strides.len() == 2 {
        if bias_shape[0] == 1 {
            [0, bias_strides[1] as i64]
        } else if bias_shape[0] == (m as i64) {
            [bias_strides[0] as i64, bias_strides[1] as i64]
        } else {
            panic!(
                "for 2D bias, first dim should be 1 or equal to m({}), but got shape({:?})",
                m,
                bias_shape
            );
        }
    } else {
        panic!("bias must be 1D or 2D tensor, got {}", bias_strides.len());
    };
    let bias_size = bias_shape.iter().product::<i64>();
    if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
        check_out_layout(lhs_layout, rhs_layout, out_layout)?;
        let post_op = post_op.map(|f| {
            let boxed: Box<dyn (Fn(T, usize, usize) -> T) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let post_op_vec = post_op_vec.map(|f| {
            let boxed: Box<dyn (Fn(T::Vec, usize, usize) -> T::Vec) + Send + Sync> = Box::new(f);
            unsafe { std::mem::transmute(boxed) }
        });
        let lhs_ptr = lhs;
        let rhs_ptr = rhs;
        let res_ptr = out;
        hpt_matmul::addmm::<T>(
            (lhs_ptr.ptr as *const T, lhs_ptr.len()),
            (rhs_ptr.ptr as *const T, rhs_ptr.len()),
            (res_ptr.ptr as *mut T, res_ptr.len()),
            (bias.ptr as *const T, bias_size),
            n,
            m,
            k,
            [
                lhs_layout.strides()[lhs_layout.ndim() - 2],
                lhs_layout.strides()[lhs_layout.ndim() - 1],
            ],
            [
                rhs_layout.strides()[rhs_layout.ndim() - 2],
                rhs_layout.strides()[rhs_layout.ndim() - 1],
            ],
            [
                out_layout.strides()[out_layout.ndim() - 2],
                out_layout.strides()[out_layout.ndim() - 1],
            ],
            bias_strides,
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec
        );
        Ok(())
    } else {
        fn extract_batch_and_matrix_dims(
            layout: &Layout
        ) -> Result<(Vec<i64>, i64, i64), TensorError> {
            let shape = layout.shape();
            let ndim = shape.len();
            if ndim < 2 {
                panic!("Tensor must have at least 2 dimensions, got {}", ndim);
            }
            let matrix_dims = &shape[ndim - 2..];
            let batch_dims = &shape[..ndim - 2];

            Ok((batch_dims.to_vec(), matrix_dims[0], matrix_dims[1]))
        }
        let (batch_dims_lhs, m, k_lhs) = extract_batch_and_matrix_dims(lhs_layout)?;
        let (batch_dims_rhs, k_rhs, n) = extract_batch_and_matrix_dims(rhs_layout)?;
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
        let broacasted_lhs = lhs_layout.broadcast_to(&Shape::from(&new_lhs_shape))?;
        let broacasted_rhs = rhs_layout.broadcast_to(&Shape::from(&new_rhs_shape))?;
        let batch_size = broadcast_batch_shape.iter().product::<i64>() as usize;
        let mut output_shape = broadcast_batch_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        check_out_layout(lhs_layout, rhs_layout, out_layout)?;

        let lhs_strides: &[i64] = &broacasted_lhs.strides()[broacasted_lhs.strides().len() - 2..];
        let rhs_strides: &[i64] = &broacasted_rhs.strides()[broacasted_rhs.strides().len() - 2..];
        let out_strides: &[i64] = &out_layout.strides()[out_layout.ndim() - 2..];

        let res_gp = dispatch_map_global_idx(out_layout);
        let lhs_gp = dispatch_map_global_idx(&broacasted_lhs);
        let rhs_gp = dispatch_map_global_idx(&broacasted_rhs);
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
                let lhs_ptr = lhs + lhs_offset;
                let rhs_ptr = rhs + rhs_offset;
                let res_ptr = out + out_offset;
                hpt_matmul::addmm::<T>(
                    (lhs_ptr.cast::<T>().ptr as *const T, lhs_ptr.len()),
                    (rhs_ptr.cast::<T>().ptr as *const T, rhs_ptr.len()),
                    (res_ptr.cast::<T>().ptr as *mut T, res_ptr.len()),
                    (bias.cast::<T>().ptr as *const T, bias_size),
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

        Ok(())
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        let mut res = matmul_prepare(self, rhs, None)?;
        self.matmul_(&rhs, &mut res)
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
                let res = matmul_prepare(self, rhs, None)?;
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
                            self.ptr::<T>(),
                            self.layout(),
                            rhs.ptr::<T>(),
                            rhs.layout(),
                            res.ptr::<T>(),
                            res.layout(),
                            current_num_threads(),
                            None,
                            None::<fn(T, usize, usize) -> T>,
                            None::<fn(InVec, usize, usize) -> InVec>,
                        )?,
                        (true, false) => {
                            matmul_with_out(
                                self.ptr::<T>(),
                                self.layout(),
                                rhs.ptr::<T>(),
                                rhs.layout(),
                                res.ptr::<T>(),
                                res.layout(),
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
                            )?
                        }
                        (false, true) => matmul_with_out(
                            self.ptr::<T>(),
                            self.layout(),
                            rhs.ptr::<T>(),
                            rhs.layout(),
                            res.ptr::<T>(),
                            res.layout(),
                            current_num_threads(),
                            None,
                            Some(move |inp: T, _, _| inp._mul(alpha_casted)),
                            Some(move |inp: InVec, _, _| inp * alpha_vec),
                        )?,
                        (false, false) => {
                            matmul_with_out(
                                self.ptr::<T>(),
                                self.layout(),
                                rhs.ptr::<T>(),
                                rhs.layout(),
                                res.ptr::<T>(),
                                res.layout(),
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
                            )?
                        }
                    }
                } else {
                    if alpha == 1.0 {
                        matmul_with_out(
                            self.ptr::<T>(),
                            self.layout(),
                            rhs.ptr::<T>(),
                            rhs.layout(),
                            res.ptr::<T>(),
                            res.layout(),
                            current_num_threads(),
                            None,
                            None::<fn(T, usize, usize) -> T>,
                            None::<fn(InVec, usize, usize) -> InVec>,
                        )?
                    } else {
                        let alpha_casted: T = alpha.cast();
                        let alpha_vec = InVec::splat(alpha_casted);
                        matmul_with_out(
                            self.ptr::<T>(),
                            self.layout(),
                            rhs.ptr::<T>(),
                            rhs.layout(),
                            res.ptr::<T>(),
                            res.layout(),
                            current_num_threads(),
                            None,
                            Some(move |inp: T, _, _| inp._mul(alpha_casted)),
                            Some(move |inp: InVec, _, _| inp * alpha_vec),
                        )?
                    }
                }
                Ok(res)
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
        let res = matmul_prepare(self, rhs, Some(out.clone()))?;
        macro_rules! matmul {
            ($dtype:ty) => {
                {
                    matmul_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                        self.ptr::<$dtype>(),
                        self.layout(),
                        rhs.ptr::<$dtype>(),
                        rhs.layout(),
                        res.ptr::<$dtype>(),
                        res.layout(),
                        current_num_threads(),
                        None,
                        None,
                        None
                    )?;
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
        Ok(res)
    }

    pub fn addmm(&self, rhs: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
        let mut res = matmul_prepare(self, rhs, None)?;
        self.addmm_(&rhs, bias, &mut res)
    }

    pub fn addmm_(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: &mut Tensor
    ) -> Result<Tensor, TensorError> {
        let res = matmul_prepare(self, rhs, Some(out.clone()))?;
        macro_rules! matmul {
            ($dtype:ty) => {
                {
                    addmm_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                        self.ptr::<$dtype>(),
                        self.layout(),
                        rhs.ptr::<$dtype>(),
                        rhs.layout(),
                        bias.ptr::<$dtype>(),
                        bias.layout(),
                        res.ptr::<$dtype>(),
                        res.layout(),
                        current_num_threads(),
                        None,
                        None,
                        None
                    )?;
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
        Ok(res)
    }
}

pub(crate) fn addmm_prepacked_(
    lhs: Pointer<u8>,
    lhs_layout: &Layout,
    rhs: Pointer<u8>,
    rhs_layout: &Layout,
    bias: Pointer<u8>,
    bias_layout: &Layout,
    out: Pointer<u8>,
    out_layout: &Layout,
    num_threads: usize,
    dtype: DType,
    prepacked_rhs: Option<Arc<hpt_matmul::PrePackedRhs>>
) -> Result<(), TensorError> {
    macro_rules! matmul {
        ($dtype:ty) => {
            addmm_with_out::<$dtype, fn($dtype, usize, usize) -> $dtype, fn(<$dtype as TypeCommon>::Vec, usize, usize) -> <$dtype as TypeCommon>::Vec>(
                lhs.cast::<$dtype>(),
                lhs_layout,
                rhs.cast::<$dtype>(),
                rhs_layout,
                bias.cast::<$dtype>(),
                bias_layout,
                out.cast::<$dtype>(),
                out_layout,
                num_threads,
                prepacked_rhs,
                None,
                None
            )?
        };
    }
    match dtype {
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
        _ => unimplemented!("unsupported dtype {:?}", dtype),
    }
    Ok(())
}
