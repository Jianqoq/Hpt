use gemm_common::cache::DivCeil;
use hpt_common::{
    Pointer,
    error::{base::TensorError, shape::ShapeError},
    shape::shape_utils::predict_broadcast_shape,
};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{Device, Tensor, current_num_threads, physical_cores};

#[cfg(feature = "bf16")]
use super::matmul_mp::bf16_matmul_mp_no_block_info;
#[cfg(feature = "f16")]
use super::matmul_mp::f16_matmul_mp_no_block_info;
#[cfg(feature = "bf16")]
use super::matmul_mp_post::bf16_matmul_mp_post_no_block_info;
#[cfg(feature = "f16")]
use super::matmul_mp_post::f16_matmul_mp_post_no_block_info;
use super::{
    common::matmul_prepare,
    matmul_post::matmul_post_template_no_block_info,
    microkernel_trait::MatmulMicroKernel,
    utils::{PrePackedRhs, kernel_params},
};

use hpt_types::{
    dtype::{ToDType, TypeCommon},
    traits::VecTrait,
    type_promote::NormalOut,
};

#[cfg(feature = "bf16")]
use half::bf16;
#[cfg(feature = "f16")]
use half::f16;

use hpt_types::into_scalar::Cast;

fn matmul_template_no_block_info_prepack_rhs<T>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    device: Device,
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + MatmulMicroKernel + ToDType,
{
    let (nr, mr) = if m > 1 {
        (T::get_max_nr() * T::Vec::SIZE, T::get_max_mr())
    } else {
        (T::get_max_horizontal_nr() * T::Vec::SIZE, 1)
    };
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
    super::template::prepack_b::<T>(
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
    )
}

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
    prepack_rhs: Option<PrePackedRhs>,
) where
    T: CommonBounds + MatmulMicroKernel,
{
    let (nr, mr) = if m > 1 {
        (T::get_max_nr() * T::Vec::SIZE, T::get_max_mr())
    } else {
        (T::get_max_horizontal_nr() * T::Vec::SIZE, 1)
    };
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
        prepack_rhs,
        |_, _, _| T::ZERO,
        |_, _, _| T::Vec::splat(T::ZERO),
    );
}

pub(crate) fn matmul<T, F1, F2>(
    lhs: Pointer<T>,
    rhs: Pointer<T>,
    out: Pointer<T>,
    lhs_strides: &[i64],
    rhs_strides: &[i64],
    out_strides: &[i64],
    lhs_shape: &[i64],
    rhs_shape: &[i64],
    mut threads: usize,
    prepacked_rhs: Option<PrePackedRhs>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    threads = threads.min(crate::physical_cores());
    let lhs_cs = lhs_strides[lhs_strides.len() - 1];
    let lhs_rs = lhs_strides[lhs_strides.len() - 2];
    let dst_cs = out_strides[out_strides.len() - 2];
    let rhs_cs = rhs_strides[rhs_strides.len() - 1];
    let rhs_rs = rhs_strides[rhs_strides.len() - 2];
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    match (post_op, post_op_vec) {
        (None, None) => match T::STR {
            #[cfg(feature = "f16")]
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
                        prepacked_rhs,
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
                        prepacked_rhs,
                    )
                }
            }
            #[cfg(feature = "bf16")]
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
                prepacked_rhs,
            ),
            _ => {
                if threads == 1 {
                    hpt_matmul::single_thread_matmul::<T>(
                        lhs,
                        rhs,
                        out,
                        lhs_rs,
                        rhs_rs,
                        dst_cs,
                        lhs_cs,
                        rhs_cs,
                        threads,
                        prepacked_rhs,
                    );
                } else {
                    matmul_template_no_block_info(
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
                        prepacked_rhs,
                    );
                }
        }
    },
        (Some(post_op), Some(post_op_vec)) => match T::STR {
            #[cfg(feature = "f16")]
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
                        prepacked_rhs,
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
                        prepacked_rhs,
                        post_op,
                        post_op_vec,
                    )
                }
            }
            #[cfg(feature = "bf16")]
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
                prepacked_rhs,
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
                prepacked_rhs,
            ),
        },
        _ => panic!("post_op and post_op_vec must be both Some or both None"),
    }
}

#[duplicate::duplicate_item(
    func_name    T    ;
    [new_matmul_f32] [f32];
)]
pub(crate) fn func_name<F1, F2>(
    lhs: Pointer<T>,
    rhs: Pointer<T>,
    out: Pointer<T>,
    lhs_strides: &[i64],
    rhs_strides: &[i64],
    out_strides: &[i64],
    lhs_shape: &[i64],
    rhs_shape: &[i64],
    mut threads: usize,
    prepacked_rhs: Option<hpt_matmul::PrePackedRhs>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as TypeCommon>::Vec, usize, usize) -> <T as TypeCommon>::Vec + Clone + Send + Sync + 'static,
{
    threads = threads.min(crate::physical_cores());
    let lhs_cs = lhs_strides[lhs_strides.len() - 1];
    let lhs_rs = lhs_strides[lhs_strides.len() - 2];
    let dst_cs = out_strides[out_strides.len() - 2];
    let rhs_cs = rhs_strides[rhs_strides.len() - 1];
    let rhs_rs = rhs_strides[rhs_strides.len() - 2];
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    match (post_op, post_op_vec) {
        (None, None) => match T::STR {
            _ => hpt_matmul::matmul(
                lhs.ptr as *const T,
                rhs.ptr as *const T,
                out.ptr,
                m,
                n,
                k,
                lhs_rs,
                rhs_rs,
                dst_cs,
                lhs_cs,
                rhs_cs,
                threads,
                prepacked_rhs,
            ),
        },
        (Some(post_op), Some(post_op_vec)) => match T::STR {
            _ => hpt_matmul::matmul_with_post(
                lhs.ptr as *const T,
                rhs.ptr as *const T,
                out.ptr,
                m,
                n,
                k,
                lhs_rs,
                rhs_rs,
                dst_cs,
                lhs_cs,
                rhs_cs,
                threads,
                prepacked_rhs,
                post_op,
                |x, m, n| unsafe {
                    let x: <T as TypeCommon>::Vec = std::mem::transmute(x);
                    std::mem::transmute(post_op_vec(x, m, n))
                },
            ),
        },
        _ => panic!("post_op and post_op_vec must be both Some or both None"),
    }
}

pub(crate) fn matmul_prepack_rhs<T>(
    rhs: Pointer<T>,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    rhs_row_stride: i64,
    lhs_shape: &[i64],
    rhs_shape: &[i64],
    threads: usize,
    device: Device,
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + MatmulMicroKernel + ToDType,
{
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;
    match T::STR {
        #[cfg(feature = "f16")]
        "f16" => {
            if cfg!(target_feature = "neon") && cfg!(target_feature = "fp16") {
                matmul_template_no_block_info_prepack_rhs::<f16>(
                    rhs.cast::<f16>(),
                    m,
                    n,
                    k,
                    rhs_row_stride,
                    lhs_col_stride,
                    rhs_col_stride,
                    threads,
                    device,
                )
            } else {
                super::matmul_mp::f16_matmul_mp_no_block_info_prepack_rhs::<f16, f32>(
                    rhs.cast::<f16>(),
                    m,
                    n,
                    k,
                    rhs_row_stride,
                    rhs_col_stride,
                    threads,
                    device,
                )
            }
        }
        #[cfg(feature = "bf16")]
        "bf16" => super::matmul_mp::bf16_matmul_mp_no_block_info_prepack_rhs::<bf16, f32>(
            rhs.cast::<bf16>(),
            m,
            n,
            k,
            rhs_row_stride,
            rhs_col_stride,
            threads,
            device,
        ),
        _ => matmul_template_no_block_info_prepack_rhs(
            rhs,
            m,
            n,
            k,
            rhs_row_stride,
            lhs_col_stride,
            rhs_col_stride,
            threads,
            device,
        ),
    }
}

#[track_caller]
pub(crate) fn matmul_with_out<T, F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    out: Option<Tensor>,
    mut threads: usize,
    prepacked_rhs: Option<PrePackedRhs>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) -> std::result::Result<Tensor, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    threads = threads.min(num_cpus::get_physical());
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
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec,
        );
        Ok(c)
    } else {
        fn extract_batch_and_matrix_dims(
            tensor: &Tensor,
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
        if let Some(_) = prepacked_rhs {
            panic!("prepacked_rhs is not supported for batch matmul");
        } else {
            (0..batch_size).into_par_iter().for_each(|i| {
                let i = i as i64;
                let lhs_offset = lhs_gp(i * m * k);
                let rhs_offset = rhs_gp(i * k * n);
                let out_offset = res_gp(i * m * n);
                matmul(
                    lhs.data.cast::<T>() + lhs_offset,
                    rhs.data.cast::<T>() + rhs_offset,
                    result.data.cast::<T>() + out_offset,
                    lhs_strides,
                    rhs_strides,
                    out_strides,
                    &[m, k],
                    &[k, n],
                    1,
                    None,
                    post_op.clone(),
                    post_op_vec.clone(),
                );
            });
        }

        Ok(result)
    }
}

#[track_caller]
pub(crate) fn new_matmul_with_out<T, F1, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    out: Option<Tensor>,
    mut threads: usize,
    prepacked_rhs: Option<PrePackedRhs>,
    post_op: Option<F1>,
    post_op_vec: Option<F2>,
) -> Result<Tensor, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    threads = threads.min(num_cpus::get_physical());
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
            threads,
            prepacked_rhs,
            post_op,
            post_op_vec,
        );
        Ok(c)
    } else {
        panic!("Tensor must have at least 2 dimensions, got {}", lhs.shape().len());
    }
}

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, TensorError> {
        self._matmul(rhs, current_num_threads(), None)
    }
    pub(crate) fn _matmul(
        &self,
        rhs: &Tensor,
        num_threads: usize,
        prepacked_rhs: Option<PrePackedRhs>,
    ) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {
                matmul_with_out(
                    self,
                    rhs,
                    None,
                    num_threads,
                    prepacked_rhs,
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

    pub fn gemm(
        &self,
        rhs: &Tensor,
        bias: Option<&Tensor>,
        alpha: f64,
        beta: f64,
    ) -> Result<Tensor, TensorError> {
        macro_rules! matmul {
            ($dtype:ty) => {{
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
            }};
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
        matmul_with_out(
            self,
            rhs,
            None,
            current_num_threads(),
            None,
            Some(post_op),
            Some(post_op_vec),
        )
    }
    pub(crate) fn _matmul_f32_<F1, F2>(
        &self,
        rhs: &Tensor,
        out: Option<Tensor>,
        threads: usize,
        pre_packed_rhs: Option<hpt_matmul::PrePackedRhs>,
        post_op: Option<F1>,
        post_op_vec: Option<F2>,
    ) -> Result<Tensor, TensorError>
    where
        F1: Fn(f32, usize, usize) -> f32 + Clone + Send + Sync + 'static,
        F2: Fn(<f32 as TypeCommon>::Vec, usize, usize) -> <f32 as TypeCommon>::Vec + Clone + Send + Sync + 'static,
    {
        let c = matmul_prepare(self, &rhs, out)?;
        new_matmul_f32(
            self.data.cast::<f32>(),
            rhs.data.cast::<f32>(),
            c.data.cast::<f32>(),
            self.strides(),
            rhs.strides(),
            c.strides(),
            self.shape(),
            rhs.shape(),
            threads.min(physical_cores()),
            pre_packed_rhs,
            post_op,
            post_op_vec
        );
        Ok(c)
    }
    pub(crate) fn _addmm_f32_(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: Option<Tensor>,
        threads: usize,
        pre_packed_rhs: Option<hpt_matmul::PrePackedRhs>,
    ) -> Result<Tensor, TensorError>
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
        let bias_ptr = bias.ptr::<f32>();
        self._matmul_f32_(
            rhs,
            out,
            threads,
            pre_packed_rhs,
            Some(move |inp, m, n| {
                bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs]._add(inp)
            }),
            Some(move |inp, m, n| {
                let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                (bias_ptr + offset)
                    .cast::<<f32 as TypeCommon>::Vec>()
                    .read_unaligned()
                    + inp
            }),
        )
    }
    pub(crate) fn _matmul_post<T, F1, F2>(
        &self,
        rhs: &Tensor,
        num_threads: usize,
        out: Option<Tensor>,
        prepacked_rhs: Option<PrePackedRhs>,
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
            out,
            num_threads,
            prepacked_rhs,
            Some(post_op),
            Some(post_op_vec),
        )
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
            current_num_threads(),
            None,
            Some(post_op),
            Some(post_op_vec),
        )
    }
    pub fn addmm(&self, rhs: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
        self._addmm(rhs, bias, None, current_num_threads(), None)
    }
    pub fn addmm_(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        self._addmm(rhs, bias, Some(out.clone()), current_num_threads(), None)
    }

    pub(crate) fn _addmm(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        out: Option<Tensor>,
        num_threads: usize,
        prepacked_rhs: Option<PrePackedRhs>,
    ) -> Result<Tensor, TensorError> {
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
                self._matmul_post::<$dtype, _, _>(
                    rhs,
                    num_threads,
                    out,
                    prepacked_rhs,
                    move |inp, m, n| {
                        bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs]._add(inp)
                    },
                    move |inp, m, n| {
                        let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                        (bias_ptr + offset)
                            .cast::<<$dtype as TypeCommon>::Vec>()
                            .read_unaligned()
                            + inp
                    },
                )
            }};
        }

        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => addmm!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => addmm!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => addmm!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => addmm!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => addmm!(bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => addmm!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => addmm!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => addmm!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => addmm!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => addmm!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => addmm!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => addmm!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => addmm!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }

    pub fn addmm_post<T, F1, F2>(
        &self,
        rhs: &Tensor,
        bias: &Tensor,
        post_op: F1,
        post_op_vec: F2,
    ) -> Result<Tensor, TensorError>
    where
        T: CommonBounds + MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
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

        let bias_ptr = bias.ptr::<T>();
        self.matmul_post::<T, _, _>(
            rhs,
            move |inp, m, n| {
                post_op(
                    bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs]._add(inp),
                    m,
                    n,
                )
            },
            move |inp, m, n| {
                let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                post_op_vec(
                    (bias_ptr + offset)
                        .cast::<<T as TypeCommon>::Vec>()
                        .read_unaligned()
                        ._add(inp),
                    m,
                    n,
                )
            },
        )
    }
}
