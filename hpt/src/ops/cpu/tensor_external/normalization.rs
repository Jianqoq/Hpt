use hpt_allocator::traits::{ AllocatorOutputRetrive, Allocator };
use hpt_common::error::base::TensorError;
use hpt_traits::ops::normalization::NormalizationOps;
use hpt_traits::CommonBounds;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use crate::Tensor;
use crate::Cpu;

impl<T, const DEVICE: usize, Al> NormalizationOps
    for Tensor<T, Cpu, DEVICE, Al>
    where T: CommonBounds, Al: Allocator + Send + Sync, Al::Output: AllocatorOutputRetrive
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;

    type InplaceOutput = Tensor<T, Cpu, DEVICE, Al>;

    type OutputMeta = T;

    fn layernorm<S>(
        &self,
        _: S,
        _: Option<&Self::Output>,
        _: Option<&Self::Output>,
        _: Self::OutputMeta
    ) -> Result<Self::Output, TensorError> {
        todo!()
    }

    fn softmax(&self, _: i64) -> Result<Self::Output, TensorError> {
        todo!()
    }

    fn log_softmax(&self, _: i64) -> Result<Self::Output, TensorError> {
        todo!()
    }

    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError>
        where f64: Cast<Self::OutputMeta>,
        Self::OutputMeta: NormalOut<bool, Output = Self::OutputMeta>
    {
        Ok(self.inner.dropout(rate)?.into())
    }
}
