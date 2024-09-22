use tensor_traits::{CommonBounds, NormalReduce, TensorInfo};
use tensor_types::{
    into_scalar::IntoScalar,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};

use crate::{backend::Cpu, tensor_base::_Tensor};

use super::unary::FloatBinaryType;
use tensor_types::dtype::TypeCommon;

impl<T> _Tensor<T, Cpu> {
    /// Applies Lp normalization along a specified axis.
    ///
    /// This method normalizes the tensor along the specified axis using the Lp norm, where `p` is the order
    /// of the norm. The Lp norm is defined as the p-th root of the sum of the absolute values of the elements
    /// raised to the power of `p`. The result is a tensor where the elements along the given axis are normalized
    /// according to the Lp norm.
    ///
    /// # Arguments
    ///
    /// * `p` - The order of the norm (Lp norm). Common values include:
    ///   - `1` for L1 normalization (sum of absolute values),
    ///   - `2` for L2 normalization (Euclidean norm),
    ///   - Other values for general Lp norms.
    /// * `axis` - The axis along which to apply the normalization. The elements along this axis will be normalized.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with Lp normalization applied along the specified axis.
    pub fn lp_normalization(&self, p: u8, axis: i64) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
    where
        T: CommonBounds + IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output:
            CommonBounds + FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<T::Vec, Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
            + FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
        T: FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
        T::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
        T::Vec: FloatOutBinary<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
        T::Vec: FloatOutBinary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
    {
        let axis = if axis < 0 {
            (self.shape().len() as i64) + axis
        } else {
            axis
        };
        match p {
            1 => {
                let norm = self.reducel1(axis, false)?;
                Ok(self / norm)
            }
            2 => {
                let norm = self.reducel2(axis, false)?;
                Ok(self / norm)
            }
            _ => {
                unimplemented!();
            }
        }
    }
}
