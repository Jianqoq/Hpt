use tensor_traits::tensor::{ CommonBounds, TensorAlloc, TensorInfo };
use crate::iterator_traits::StridedIterator;
use crate::{ iterator_traits::IterGetSet, strided_map_mut::StridedMapMut };


pub mod strided_map_simd {
    use tensor_traits::{ CommonBounds, TensorAlloc, TensorInfo };
    use tensor_types::dtype::TypeCommon;
    use crate::iterator_traits::StridedIteratorSimd;
    use crate::{
        iterator_traits::IterGetSetSimd,
        strided_map_mut::strided_map_mut_simd::StridedMapMutSimd,
    };

    #[derive(Clone)]
    pub struct StridedMapSimd<'a, I, T: 'a, F, F2> where I: 'a + IterGetSetSimd<Item = T> {
        pub(crate) iter: I,
        pub(crate) f: F,
        pub(crate) f2: F2,
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, I: 'a + IterGetSetSimd<Item = T>, T: 'a, F, F2> StridedMapSimd<'a, I, T, F, F2> {
        pub fn collect<U>(self) -> U
            where
                F: Fn(T) -> U::Meta + Sync + Send + 'a,
                F2: Fn(
                    <I as IterGetSetSimd>::SimdItem
                ) -> <<U as TensorAlloc>::Meta as TypeCommon>::Vec +
                    Sync +
                    Send +
                    'a,
                U: Clone + TensorInfo<U::Meta> + TensorAlloc,
                <I as IterGetSetSimd>::Item: Send,
                <U as TensorAlloc>::Meta: CommonBounds,
                <<U as TensorAlloc>::Meta as TypeCommon>::Vec: Send
        {
            let res = U::_empty(self.iter.shape().clone()).unwrap();
            let strided_mut = StridedMapMutSimd::new(res.clone());
            let zip = strided_mut.zip(self.iter);
            zip.for_each(
                |(x, y)| {
                    *x = (self.f)(y);
                },
                |(x, y)| {
                    *x = (self.f2)(y);
                }
            );
            res
        }
    }
}

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
