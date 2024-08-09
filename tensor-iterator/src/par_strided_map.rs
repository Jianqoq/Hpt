use rayon::iter::{ plumbing::UnindexedProducer, ParallelIterator };
use tensor_traits::tensor::{ CommonBounds, TensorAlloc, TensorInfo };

use crate::{ iterator_traits::IterGetSet, par_strided_map_mut::ParStridedMapMut };

#[derive(Clone)]
pub struct ParStridedMap<'a, I, T: 'a, F>
    where I: UnindexedProducer<Item = T> + 'a + IterGetSet<Item = T> + ParallelIterator {
    pub(crate) iter: I,
    pub(crate) f: F,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<
    'a,
    I: UnindexedProducer<Item = T> + 'a + IterGetSet<Item = T> + ParallelIterator,
    T: 'a,
    F
> ParStridedMap<'a, I, T, F> {
    pub fn collect<U>(self) -> U
        where
            F: Fn(T) -> U::Meta + Sync + Send + 'a,
            U: Clone + TensorInfo<U::Meta> + TensorAlloc,
            <I as IterGetSet>::Item: Send,
            <U as TensorAlloc>::Meta: CommonBounds
    {
        let res = U::_empty(self.iter.shape().clone()).unwrap();
        let strided_mut = ParStridedMapMut::new(res.clone());
        let zip = strided_mut.zip(self.iter);
        zip.for_each(|(x, y)| {
            *x = (self.f)(y);
        });
        res
    }
}
