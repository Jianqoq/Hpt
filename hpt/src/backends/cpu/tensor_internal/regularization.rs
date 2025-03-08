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
use hpt_types::{traits::VecTrait, type_promote::NormalOut};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    traits::SimdSelect,
    type_promote::{Cmp, NormalOutUnary, SimdCmp},
};
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
                || rand::thread_rng(),
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
        let neg_lambda = lambda._neg();
        let neg_lambda_vec = lambda_vec._neg();
        let bias_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(bias);

        Ok(self
            .par_iter_simd()
            .strided_map_simd(
                |(x, y)| {
                    *x = if y._gt(lambda) {
                        y._sub(bias)
                    } else if y._lt(neg_lambda) {
                        y._add(bias)
                    } else {
                        T::ZERO
                    };
                },
                |(x, y)| {
                    let gt_mask = y._gt(lambda_vec);
                    let lt_mask = y._lt(neg_lambda_vec);
                    let sub_bias = y._sub(bias_vec);
                    let add_bias = y._add(bias_vec);
                    let zero = T::Vec::splat(T::ZERO);
                    let res = gt_mask.select(sub_bias, zero);
                    x.write_unaligned(lt_mask.select(add_bias, res));
                },
            )
            .collect())
    }
}
