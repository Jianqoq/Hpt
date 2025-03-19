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
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    traits::SimdSelect,
    type_promote::{Cmp, NormalOutUnary, SimdCmp},
};
use hpt_types::{traits::VecTrait, type_promote::NormalOut};
use rand_distr::Distribution;
use rayon::iter::ParallelIterator;

impl<T, const DEVICE: usize, A> RegularizationOps for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds + Cmp<Output = bool>,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
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
                || rand::rng(),
                |rng, (ret, val)| {
                    let mask: Self::OutputMeta = bernoli.sample(rng).cast();
                    *ret = val._mul(mask)._mul(scale);
                },
            );
        Ok(ret)
    }

    fn shrinkage(
        &self,
        bias: Self::OutputMeta,
        lambda: Self::OutputMeta,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        let lambda_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(lambda);
        let bias_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(bias);

        Ok(self
            .par_iter_simd()
            .strided_map_simd(
                |(x, y)| {
                    let shifted = y._sub(bias);
                    let abs_shifted = shifted._abs();
                    let thresholded = abs_shifted._sub(lambda)._max(T::ZERO);
                    *x = shifted._signum()._mul(thresholded);
                },
                |(x, y)| {
                    let shifted = y._sub(bias_vec);
                    let abs_shifted = shifted._abs();
                    let sign_shifted = shifted._signum();

                    let thresholded = abs_shifted._sub(lambda_vec)._max(T::Vec::splat(T::ZERO));

                    x.write_unaligned(sign_shifted._mul(thresholded));
                },
            )
            .collect())
    }
}
