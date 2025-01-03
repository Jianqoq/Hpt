use tensor_common::axis::{ process_axes, Axis };
use tensor_common::err_handler::TensorError;
use tensor_traits::{ CommonBounds, IndexReduce, TensorInfo };
use tensor_types::type_promote::{ Cmp, NormalOut };

use crate::{ ops::cpu::reduce::{ argmax, argmin }, tensor_base::_Tensor };

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>> IndexReduce for _Tensor<T> {
    type Output = _Tensor<i64>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmax(self, axes, 0, keep_dims, None)
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmin(self, axes, 0, keep_dims, None)
    }
}
