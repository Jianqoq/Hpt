use crate::{backend::Cuda, tensor_base::_Tensor};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DeviceRepr};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::shape::ShapeError;
use hpt_common::layout::layout::Layout;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::compare_and_pad_shapes_strides;
use hpt_common::strides::strides::Strides;
use hpt_common::{error::base::TensorError, shape::shape_utils::predict_broadcast_shape};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::{CudaType, TypeCommon};
use std::borrow::{Borrow, BorrowMut};

pub(crate) fn compute_op_and_cs(strides: &Strides, shape: &Shape) -> (cublasOperation_t, i64) {
    let ndim = strides.len();
    let lhs_strides = [strides[ndim - 2], strides[ndim - 1]];
    let (op, cs) = match lhs_strides {
        [1, 1] => (
            if shape[ndim - 2] == 1 && shape[ndim - 1] != 1 {
                // A^T
                cublasOperation_t::CUBLAS_OP_N
            } else {
                // A
                cublasOperation_t::CUBLAS_OP_T
            },
            if shape[ndim - 2] == 1 {
                // A^T
                shape[ndim - 2]
            } else {
                // A
                shape[ndim - 1]
            },
        ),
        [1, _] => (cublasOperation_t::CUBLAS_OP_N, shape[ndim - 2]),
        [_, 1] => (cublasOperation_t::CUBLAS_OP_T, shape[ndim - 1]),
        [_, _] => {
            panic!("strides is not contiguous");
        }
    };
    (op, cs)
}

pub(crate) fn compute_gemm_config<T: CudaType + TypeCommon>(
    lhs_strides: &Strides,
    rhs_strides: &Strides,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    res_shape: &Shape,
    lhs: CudaSlice<T>,
    rhs: CudaSlice<T>,
) -> (GemmConfig<T>, CudaSlice<T>, CudaSlice<T>) {
    let (lhs_op, lhs_cs) = compute_op_and_cs(lhs_strides, lhs_shape);
    let (rhs_op, rhs_cs) = compute_op_and_cs(rhs_strides, rhs_shape);

    let (lhs_op, rhs_op) = match (lhs_op, rhs_op) {
        (cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T)
        | (cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N) => (lhs_op, rhs_op),
        (cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T) => (
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
        ),
        (cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N) => (
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_T,
        ),
        _ => panic!("Unsupported operation"),
    };
    let ndim = res_shape.len();
    let (m, n, dst_cs, lhs, lhs_cs, rhs, rhs_cs) = gemm_input_transpose(
        lhs_shape[ndim - 2] as usize,
        rhs_shape[ndim - 1] as usize,
        res_shape[ndim - 1] as isize,
        lhs,
        lhs_cs as isize,
        rhs,
        rhs_cs as isize,
    );
    let config = GemmConfig {
        transa: lhs_op,
        transb: rhs_op,
        m: m as i32,
        n: n as i32,
        k: lhs_shape[ndim - 1] as i32,
        alpha: T::ONE,
        lda: lhs_cs as i32,
        ldb: rhs_cs as i32,
        beta: T::ZERO,
        ldc: dst_cs as i32,
    };
    (config, lhs, rhs)
}

pub(crate) fn gemm_input_transpose<T: CudaType>(
    m: usize,
    n: usize,
    dst_rs: isize,
    lhs: CudaSlice<T>,
    lhs_rs: isize,
    rhs: CudaSlice<T>,
    rhs_rs: isize,
) -> (
    usize,
    usize,
    isize,
    CudaSlice<T>,
    isize,
    CudaSlice<T>,
    isize,
) {
    (n, m, dst_rs, rhs, rhs_rs, lhs, lhs_rs)
}

#[track_caller]
pub(crate) fn matmul_with_out<T, O, const CUDA_DEVICE: usize, Al>(
    lhs: &_Tensor<T, Cuda, CUDA_DEVICE, Al>,
    rhs: &_Tensor<T, Cuda, CUDA_DEVICE, Al>,
    out: Option<O>,
) -> std::result::Result<_Tensor<T, Cuda, CUDA_DEVICE, Al>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType,
    CudaBlas: Gemm<T>,
    O: Borrow<_Tensor<T, Cuda, CUDA_DEVICE, Al>> + BorrowMut<_Tensor<T, Cuda, CUDA_DEVICE, Al>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res: _Tensor<T, Cuda, CUDA_DEVICE, Al> = if let Some(out) = out {
            if out.borrow().size() == ((lhs.shape()[0] * rhs.shape()[1]) as usize)
                && out.borrow().parent().is_none()
            {
                let out: _Tensor<T, Cuda, CUDA_DEVICE, Al> = out.borrow().clone();
                let mut slice = unsafe {
                    out.device()
                        .upgrade_device_ptr(out.ptr().ptr as u64, out.size())
                };
                out.device()
                    .htod_sync_copy_into(&vec![T::ZERO; out.size() as usize], &mut slice)?;
                slice.leak();
                out
            } else {
                _Tensor::<T, Cuda, CUDA_DEVICE, Al>::empty(vec![lhs.shape()[0], rhs.shape()[1]])?
            }
        } else {
            _Tensor::<T, Cuda, CUDA_DEVICE, Al>::empty(vec![lhs.shape()[0], rhs.shape()[1]])?
        };
        let cublas = cudarc::cublas::CudaBlas::new(res.device())?;
        // let k = lhs.shape()[1] as i32;
        let mut slice = unsafe {
            res.device()
                .upgrade_device_ptr::<T>(res.ptr().ptr as u64, res.size())
        };
        let a_slice = unsafe {
            lhs.device()
                .upgrade_device_ptr::<T>(lhs.ptr().ptr as u64, lhs.size())
        };
        let b_slice = unsafe {
            rhs.device()
                .upgrade_device_ptr::<T>(rhs.ptr().ptr as u64, rhs.size())
        };
        let (config, a_slice, b_slice) = compute_gemm_config(
            &lhs.strides(),
            &rhs.strides(),
            &lhs.shape(),
            &rhs.shape(),
            &res.shape(),
            a_slice,
            b_slice,
        );
        unsafe { cublas.gemm(config, &a_slice, &b_slice, &mut slice)? };
        slice.leak();
        a_slice.leak();
        b_slice.leak();
        Ok(res)
    } else {
        let (longer_shape, padded_short_shape, longer_strides, padded_short_strides) =
            compare_and_pad_shapes_strides(
                &lhs.shape(),
                &rhs.shape(),
                &lhs.strides(),
                &rhs.strides(),
            );
        let a_shape;
        let b_shape;
        let a_strides;
        let b_strides;
        if lhs.shape().len() > rhs.shape().len() {
            a_shape = longer_shape;
            b_shape = padded_short_shape;
            a_strides = longer_strides;
            b_strides = padded_short_strides;
        } else {
            a_shape = padded_short_shape;
            b_shape = longer_shape;
            a_strides = padded_short_strides;
            b_strides = longer_strides;
        }
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let mut res_shape =
            predict_broadcast_shape(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?
                .to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            if out.borrow().size() == (res_shape.iter().product::<i64>() as usize) {
                let out: _Tensor<T, Cuda, CUDA_DEVICE, Al> = out.borrow().clone();
                let mut slice = unsafe {
                    out.device()
                        .upgrade_device_ptr(out.ptr().ptr as u64, out.size())
                };
                out.device()
                    .htod_sync_copy_into(&vec![T::ZERO; out.size() as usize], &mut slice)?;
                slice.leak();
                out
            } else {
                _Tensor::<T, Cuda, CUDA_DEVICE, Al>::empty(res_shape)?
            }
        } else {
            _Tensor::<T, Cuda, CUDA_DEVICE, Al>::empty(res_shape)?
        };
        let cublas = cudarc::cublas::CudaBlas::new(res.device())?;
        let mut slice = unsafe {
            res.device()
                .upgrade_device_ptr(res.ptr().ptr as u64, res.size())
        };
        let a_slice = unsafe {
            lhs.device()
                .upgrade_device_ptr(lhs.ptr().ptr as u64, lhs.size())
        };
        let b_slice = unsafe {
            rhs.device()
                .upgrade_device_ptr(rhs.ptr().ptr as u64, rhs.size())
        };
        let (config, a_slice, b_slice) = compute_gemm_config(
            &lhs.strides(),
            &rhs.strides(),
            &lhs.shape(),
            &rhs.shape(),
            &res.shape(),
            a_slice,
            b_slice,
        );
        let a_layout = Layout::new(&a_shape, &a_strides);
        let b_layout = Layout::new(&b_shape, &b_strides);
        let a_batch_dim = a_shape[..a_shape.len() - 2].iter().product::<i64>();
        let b_batch_dim = b_shape[..b_shape.len() - 2].iter().product::<i64>();

        let a_new_shape = vec![
            a_batch_dim,
            a_shape[a_shape.len() - 2],
            a_shape[a_shape.len() - 1],
        ];
        let b_new_shape = vec![
            b_batch_dim,
            b_shape[b_shape.len() - 2],
            b_shape[b_shape.len() - 1],
        ];
        match (
            a_layout.is_reshape_possible(&a_new_shape),
            b_layout.is_reshape_possible(&b_new_shape),
        ) {
            (Some(a_strides), Some(b_strides)) => {
                let config = StridedBatchedConfig {
                    gemm: config,
                    batch_size: res.shape()[..res.ndim() - 2].iter().product::<i64>() as i32,
                    stride_a: b_strides[b_strides.len() - 3], // we always transpose lhs and rhs for gemm
                    stride_b: a_strides[a_strides.len() - 3],
                    stride_c: (res.shape()[res.ndim() - 2] * res.shape()[res.ndim() - 1]) as i64,
                };
                unsafe { cublas.gemm_strided_batched(config, &a_slice, &b_slice, &mut slice)? };
            }
            _ => {
                panic!("Cannot reshape a or b to match the output shape");
            }
        }
        slice.leak();
        a_slice.leak();
        b_slice.leak();
        Ok(res)
    }
}
