use tensor_traits::tensor::{ CommonBounds, TensorAlloc, TensorInfo };

use crate::{ iterator_traits::{ IterGetSet, StridedIterator }, strided_map_mut::StridedMapMut };

#[derive(Clone)]
pub struct StridedMap<'a, I, T: 'a, F> where I: 'a + IterGetSet<Item = T> {
    pub(crate) iter: I,
    pub(crate) f: F,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, I: 'a + IterGetSet<Item = T>, T: 'a, F> StridedMap<'a, I, T, F> {
    pub fn collect<U>(self) -> U
        where
            F: Fn(T) -> U::Meta + Sync + Send + 'a,
            U: Clone + TensorInfo<U::Meta> + TensorAlloc,
            <I as IterGetSet>::Item: Send,
            <U as TensorAlloc>::Meta: CommonBounds
    {
        let res = U::_empty(self.iter.shape().clone()).unwrap();
        let strided_mut = StridedMapMut::new(res.clone());
        let zip = strided_mut.zip(self.iter);
        zip.for_each(|(x, y)| {
            *x = (self.f)(y);
        });
        res
    }
}
