use tensor_common::axis::axis::{process_axes, Axis};
use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, IndexReduce, TensorInfo};
use tensor_types::type_promote::{Cmp, NormalOut};

use crate::Cpu;
use crate::{
    ops::cpu::utils::reduce::reduce::{argmax, argmin},
    tensor_base::_Tensor,
};

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>, const DEVICE: usize>
    IndexReduce for _Tensor<T, Cpu, DEVICE>
{
    type Output = _Tensor<i64, Cpu, DEVICE>;

    fn argmax<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmax(self, axes, 0, keep_dims, None)
    }

    fn argmin<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmin(self, axes, 0, keep_dims, None)
    }
}
