use crate::Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_traits::ops::regularization::RegularizationOps;
use hpt_traits::tensor::CommonBounds;
use hpt_types::{
    traits::SimdSelect,
    type_promote::{Cmp, SimdCmp},
};

impl<T, const DEVICE: usize, A> RegularizationOps for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds + Cmp<Output = bool>,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, A>;

    type OutputMeta = T;

    fn dropout(&self, rate: f64) -> Result<Self::Output, hpt_common::error::base::TensorError>
    where
        f64: hpt_types::into_scalar::Cast<Self::OutputMeta>,
        bool: hpt_types::into_scalar::Cast<Self::OutputMeta>,
        Self::OutputMeta: hpt_types::type_promote::NormalOut<bool, Output = Self::OutputMeta>,
    {
        Ok(self.inner.dropout(rate)?.into())
    }

    fn shrinkage(
        &self,
        bias: Self::OutputMeta,
        lambda: Self::OutputMeta,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self.inner.shrinkage(bias, lambda)?.into())
    }
}
