#![allow(unused)]

use std::borrow::Borrow;

use crate::backends::cuda::utils::binary::binary_normal::binary_fn_precompiled;
use crate::backends::cuda::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cuda;
use hpt_common::error::base::TensorError;
use hpt_cudakernels::CMP;
use hpt_traits::{ops::cmp::TensorCmp, tensor::CommonBounds};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};
impl<T, C, const DEVICE_ID: usize, Al> TensorCmp<T, C> for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + DeviceRepr + Cmp<C> + CudaType,
    C: CommonBounds + DeviceRepr + CudaType,
    Scalar<T>: Cmp<Scalar<C>, Output = Scalar<bool>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type RHS = Tensor<C, Cuda, DEVICE_ID, Al>;
    type Output = Tensor<bool, Cuda, DEVICE_ID, Al>;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "ne",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "eq",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "lt",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "gt",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "le",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_precompiled(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            "ge",
            &CMP,
            None::<_Tensor<bool, Cuda, DEVICE_ID, Al>>,
        )
        .unwrap();
        Ok(res.into())
    }
}
