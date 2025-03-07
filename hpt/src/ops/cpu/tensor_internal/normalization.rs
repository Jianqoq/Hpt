use hpt_allocator::traits::{ AllocatorOutputRetrive, Allocator };
use hpt_common::error::base::TensorError;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_traits::ops::normalization::NormalizationOps;
use hpt_traits::CommonBounds;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand_distr::Distribution;
use crate::{ TensorIterator, TensorCreator };
use crate::tensor_base::_Tensor;
use crate::{ Cpu, TensorInfo };
use rayon::iter::ParallelIterator;

impl<T, const DEVICE: usize, Al> NormalizationOps
    for _Tensor<T, Cpu, DEVICE, Al>
    where T: CommonBounds, Al: Allocator + Send + Sync, Al::Output: AllocatorOutputRetrive
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;

    type InplaceOutput = _Tensor<T, Cpu, DEVICE, Al>;

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
        let mut ret = _Tensor::<T, Cpu, DEVICE, Al>::empty(self.shape())?;
        let bernoli = rand_distr::Bernoulli
            ::new(rate)
            .expect("Failed to create Bernoulli distribution for dropout");
        let scale: T = (1.0 / (1.0 - rate)).cast();
        ret.par_iter_mut_simd()
            .zip(self.par_iter_simd())
            .for_each_init(
                || rand::thread_rng(),
                |rng, (ret, val)| {
                    let mask = bernoli.sample(rng);
                    *ret = val._mul(mask)._mul(scale);
                }
            );
        Ok(ret)
    }
}
