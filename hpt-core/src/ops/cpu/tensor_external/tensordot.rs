use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, TensorDot};
use hpt_types::type_promote::NormalOut;

use crate::{tensor_base::_Tensor, Tensor};

impl<A, B> TensorDot<Tensor<B>> for Tensor<A>
where
    _Tensor<A>: TensorDot<_Tensor<B>, Output = _Tensor<<A as NormalOut<B>>::Output>>,
    A: CommonBounds + NormalOut<B>,
    B: CommonBounds,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = Tensor<<A as NormalOut<B>>::Output>;

    fn tensordot<const N: usize>(
        &self,
        rhs: &Tensor<B>,
        axes: ([i64; N], [i64; N]),
    ) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.tensordot(rhs.inner.as_ref(), axes)?;
        Ok(res.into())
    }
}
