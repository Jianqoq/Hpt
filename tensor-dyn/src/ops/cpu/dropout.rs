use rand_distr::Distribution;
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };
use rayon::iter::ParallelIterator;
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };
use crate::{ backend::Cpu, tensor_base::_Tensor };

impl<T> _Tensor<T, Cpu>
    where
        T: CommonBounds + NormalOut<bool, Output = T> + NormalOut<T, Output = T>,
        f64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn dropout(&self, rate: f64) -> anyhow::Result<_Tensor<T>> {
        let ret = _Tensor::<T>::empty(self.shape())?;
        let bernoli = rand::distributions::Bernoulli::new(rate)?;
        let scale: T = (1.0 / (1.0 - rate)).into_scalar();
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
