use cudarc::{cudnn::Cudnn, driver::DeviceRepr};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::ops::pooling::{FloatOutPooling, NormalPooling};
use hpt_types::{
    dtype::{CudaType, TypeCommon},
    into_scalar::Cast,
    traits::VecTrait,
    type_promote::{FloatOutBinary, NormalOut},
};

use crate::backend::Cuda;
use crate::{
    backends::cpu::kernels::pooling::common::{adaptive_pooling_template, pooling_template},
    tensor_base::_Tensor,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_traits::tensor::CommonBounds;
impl<T, const DEVICE: usize, A> FloatOutPooling for _Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds
        + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>
        + DeviceRepr
        + CudaType,
    <T as FloatOutBinary>::Output: CommonBounds + DeviceRepr + CudaType,
    T::Vec: FloatOutBinary<
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
    >,
    bool: Cast<T>,
    i64: Cast<<T as FloatOutBinary>::Output>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<<T as FloatOutBinary>::Output, Cuda, DEVICE, A>;
    #[track_caller]
    fn avgpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }

    #[track_caller]
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}

impl<T, const DEVICE: usize, A> NormalPooling for _Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds,
    bool: Cast<T>,
    i64: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cuda, DEVICE, A>;
    #[track_caller]
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> std::result::Result<Self::Output, TensorError> {
        unimplemented!()
    }

    #[track_caller]
    fn adaptive_maxpool2d(
        &self,
        output_size: [i64; 2],
    ) -> std::result::Result<Self::Output, TensorError> {
        unimplemented!()
    }
}
