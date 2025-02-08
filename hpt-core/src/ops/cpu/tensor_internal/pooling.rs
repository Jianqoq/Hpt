use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, FloatOutPooling, NormalPooling};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    traits::VecTrait,
    type_promote::{FloatOutBinary, NormalOut},
};

use crate::{
    ops::cpu::kernels::pooling::common::{adaptive_pooling_template, pooling_template},
    tensor_base::_Tensor,
    Cpu,
};

impl<T, const DEVICE: usize> FloatOutPooling for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds
        + Cast<T>
        + NormalOut<Output = T>
        + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output:
        CommonBounds + FloatOutBinary<Output = <T as FloatOutBinary>::Output>,
    T::Vec: VecTrait<T>
        + Copy
        + Send
        + Sync
        + NormalOut<Output = T::Vec>
        + FloatOutBinary<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    bool: Cast<T>,
    i64: Cast<<T as FloatOutBinary>::Output>,
{
    type Output = _Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>;
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

impl<T, const DEVICE: usize> NormalPooling for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
    i64: Cast<T>,
{
    type Output = _Tensor<T, Cpu, DEVICE>;
    #[track_caller]
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> std::result::Result<_Tensor<T, Cpu, DEVICE>, TensorError> {
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
    ) -> std::result::Result<_Tensor<T, Cpu, DEVICE>, TensorError> {
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
