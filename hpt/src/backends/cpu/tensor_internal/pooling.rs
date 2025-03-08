use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::ops::pooling::{FloatOutPooling, NormalPooling};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    traits::VecTrait,
    type_promote::{FloatOutBinary, NormalOut},
};

use crate::backend::Cpu;
use crate::{
    backends::cpu::kernels::pooling::common::{adaptive_pooling_template, pooling_template},
    tensor_base::_Tensor,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_traits::tensor::CommonBounds;
impl<T, const DEVICE: usize, A> FloatOutPooling for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds
        + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output: CommonBounds,
    T::Vec: FloatOutBinary<
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
    >,
    bool: Cast<T>,
    i64: Cast<<T as FloatOutBinary>::Output>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE, A>;
    #[track_caller]
    fn avgpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        let kernels_shape: Shape = kernels_shape.into();
        let kernel_size: <T as FloatOutBinary>::Output = kernels_shape.size().cast();
        let kernel_size_vec =
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(kernel_size);
        pooling_template(
            self,
            &kernels_shape,
            steps,
            padding,
            dilation,
            |a: T, b: T| a._add(b),
            |a: T::Vec, b: T::Vec| a._add(b),
            |a: T| a._div(kernel_size),
            |a: T::Vec| a._div(kernel_size_vec),
        )
    }

    #[track_caller]
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        adaptive_pooling_template(
            self,
            output_size,
            |a: T, b: T| a._add(b),
            |a: T::Vec, b: T::Vec| a._add(b),
            |a: T, kernel_size: <T as FloatOutBinary>::Output| a._div(kernel_size),
            |a: T::Vec, kernel_size_vec: <<T as FloatOutBinary>::Output as TypeCommon>::Vec| {
                a._div(kernel_size_vec)
            },
        )
    }
}

impl<T, const DEVICE: usize, A> NormalPooling for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    bool: Cast<T>,
    i64: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A>;
    #[track_caller]
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> std::result::Result<Self::Output, TensorError> {
        pooling_template(
            self,
            &kernels_shape.into(),
            steps,
            padding,
            dilation,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a| a,
            |a| a,
        )
    }

    #[track_caller]
    fn adaptive_maxpool2d(
        &self,
        output_size: [i64; 2],
    ) -> std::result::Result<Self::Output, TensorError> {
        adaptive_pooling_template(
            self,
            output_size,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, _| a,
            |a, _| a,
        )
    }
}
