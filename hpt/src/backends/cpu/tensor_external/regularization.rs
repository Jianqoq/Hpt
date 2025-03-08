use crate::Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_traits::ops::regularization::RegularizationOps;
use hpt_traits::tensor::CommonBounds;

impl<T, const DEVICE: usize, A> RegularizationOps for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
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
}
