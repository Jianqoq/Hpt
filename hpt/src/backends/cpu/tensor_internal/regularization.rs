use crate::tensor_base::_Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_iterator::{iterator_traits::ParStridedIteratorSimdZip, TensorIterator};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::{
    ops::regularization::RegularizationOps,
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::into_scalar::Cast;
use rand_distr::Distribution;
use rayon::iter::ParallelIterator;

impl<T, const DEVICE: usize, A> RegularizationOps for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A>;

    type OutputMeta = T;

    fn dropout(&self, rate: f64) -> Result<Self::Output, hpt_common::error::base::TensorError>
    where
        f64: hpt_types::into_scalar::Cast<Self::OutputMeta>,
        bool: hpt_types::into_scalar::Cast<Self::OutputMeta>,
        Self::OutputMeta: hpt_types::type_promote::NormalOut<bool, Output = Self::OutputMeta>,
    {
        let mut ret = Self::Output::empty(self.shape())?;
        let bernoli = rand_distr::Bernoulli::new(rate)
            .expect("Failed to create Bernoulli distribution for dropout");
        let scale: T = (1.0 / (1.0 - rate)).cast();
        ret.par_iter_mut_simd()
            .zip(self.par_iter_simd())
            .for_each_init(
                || rand::thread_rng(),
                |rng, (ret, val)| {
                    let mask: Self::OutputMeta = bernoli.sample(rng).cast();
                    *ret = val._mul(mask)._mul(scale);
                },
            );
        Ok(ret)
    }
}
