use std::panic::Location;

use crate::CommonBounds;
use crate::{tensor_base::_Tensor, Cuda};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::DeviceRepr;
use std::borrow::{Borrow, BorrowMut};
use tensor_common::{
    err_handler::ErrHandler,
    shape_utils::{compare_and_pad_shapes, predict_broadcast_shape},
};
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn matmul_with_out<T, O, const CUDA_DEVICE: usize>(
    lhs: &_Tensor<T, Cuda, CUDA_DEVICE>,
    rhs: &_Tensor<T, Cuda, CUDA_DEVICE>,
    out: Option<O>,
) -> std::result::Result<_Tensor<T, Cuda, CUDA_DEVICE>, ErrHandler>
where
    T: CommonBounds + DeviceRepr,
    CudaBlas: Gemm<T>,
    O: Borrow<_Tensor<T, Cuda, CUDA_DEVICE>> + BorrowMut<_Tensor<T, Cuda, CUDA_DEVICE>>,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        if lhs.shape()[1] != rhs.shape()[0] {
            Err(ErrHandler::MatmulShapeMismatched(
                [lhs.shape()[0], lhs.shape()[1]],
                [rhs.shape()[0], rhs.shape()[1]],
                lhs.shape()[1],
                Location::caller(),
            )
            .into())
        } else {
            let res: _Tensor<T, Cuda, CUDA_DEVICE> = if let Some(out) = out {
                if out.borrow().size() == ((lhs.shape()[0] * rhs.shape()[1]) as usize)
                    && out.borrow().parent().is_none()
                {
                    let out: _Tensor<T, Cuda, CUDA_DEVICE> = out.borrow().clone();
                    let mut slice = unsafe {
                        out.device()
                            .upgrade_device_ptr(out.ptr().ptr as u64, out.size())
                    };
                    out.device()
                        .htod_sync_copy_into(&vec![T::ZERO; out.size() as usize], &mut slice)
                        .map_err(|e| ErrHandler::CudaHostToDeviceError(Location::caller(), e))?;
                    slice.leak();
                    out
                } else {
                    _Tensor::<T, Cuda, CUDA_DEVICE>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?
                }
            } else {
                _Tensor::<T, Cuda, CUDA_DEVICE>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?
            };
            let cublas = cudarc::cublas::CudaBlas::new(res.device())
                .map_err(|e| ErrHandler::CudaCreateCublasHandleError(Location::caller(), e))?;
            let config = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: lhs.shape()[0] as i32,
                n: rhs.shape()[1] as i32,
                k: lhs.shape()[1] as i32,
                alpha: T::ONE,
                lda: lhs.shape()[0] as i32,
                ldb: rhs.shape()[1] as i32,
                beta: T::ZERO,
                ldc: res.shape()[1] as i32,
            };
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
            unsafe {
                cublas
                    .gemm(config, &a_slice, &b_slice, &mut slice)
                    .map_err(|e| ErrHandler::CudaCublasExecuteError(Location::caller(), e))?
            };
            slice.leak();
            a_slice.leak();
            b_slice.leak();
            Ok(res)
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
        if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
            Err(ErrHandler::MatmulShapeMismatched(
                [a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1]],
                [b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1]],
                a_shape[a_shape.len() - 1],
                Location::caller(),
            )
            .into())
        } else {
            let mut res_shape = predict_broadcast_shape(
                &a_shape[..a_shape.len() - 2],
                &b_shape[..b_shape.len() - 2],
            )?
            .to_vec();
            res_shape.push(a_shape[a_shape.len() - 2]);
            res_shape.push(b_shape[b_shape.len() - 1]);
            let res = if let Some(out) = out {
                if out.borrow().size() == (res_shape.iter().product::<i64>() as usize) {
                    let out: _Tensor<T, Cuda, CUDA_DEVICE> = out.borrow().clone();
                    let mut slice = unsafe {
                        out.device()
                            .upgrade_device_ptr(out.ptr().ptr as u64, out.size())
                    };
                    out.device()
                        .htod_sync_copy_into(&vec![T::ZERO; out.size() as usize], &mut slice)
                        .map_err(|e| ErrHandler::CudaHostToDeviceError(Location::caller(), e))?;
                    slice.leak();
                    out
                } else {
                    _Tensor::<T, Cuda, CUDA_DEVICE>::zeros(res_shape)?
                }
            } else {
                _Tensor::<T, Cuda, CUDA_DEVICE>::zeros(res_shape)?
            };
            let cublas = cudarc::cublas::CudaBlas::new(res.device())
                .map_err(|e| ErrHandler::CudaCreateCublasHandleError(Location::caller(), e))?;
            let config = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_T,
                m: lhs.shape()[0] as i32,
                n: rhs.shape()[1] as i32,
                k: rhs.shape()[0] as i32,
                alpha: T::ONE,
                lda: lhs.shape()[1] as i32,
                ldb: rhs.shape()[1] as i32,
                beta: T::ZERO,
                ldc: res.shape()[1] as i32,
            };
            let config = StridedBatchedConfig {
                gemm: config,
                batch_size: res.shape()[..res.ndim() - 2].iter().product::<i64>() as i32,
                stride_a: (a_shape[a_shape.len() - 2] * a_shape[a_shape.len() - 1]) as i64,
                stride_b: (b_shape[b_shape.len() - 2] * b_shape[b_shape.len() - 1]) as i64,
                stride_c: (res.shape()[res.ndim() - 2] * res.shape()[res.ndim() - 1]) as i64,
            };
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
            unsafe {
                cublas
                    .gemm_strided_batched(config, &a_slice, &b_slice, &mut slice)
                    .map_err(|e| ErrHandler::CudaCublasExecuteError(Location::caller(), e))?
            };
            slice.leak();
            a_slice.leak();
            b_slice.leak();
            Ok(res)
        }
    }
}
