use rayon::iter::{plumbing::UnindexedProducer, ParallelIterator};
use hpt_common::shape::shape::Shape;
use hpt_traits::tensor::{CommonBounds, TensorAlloc, TensorInfo};

use crate::{
    iterator_traits::{IterGetSet, ShapeManipulator},
    par_strided_map_mut::ParStridedMapMut,
    par_strided_mut::par_strided_map_mut_simd::ParStridedMutSimd,
};

/// A module for parallel strided map iterator.
pub mod par_strided_map_simd {
    use rayon::iter::{plumbing::UnindexedProducer, ParallelIterator};
    use hpt_common::utils::simd_ref::MutVec;
    use hpt_traits::{CommonBounds, TensorAlloc, TensorInfo};
    use hpt_types::dtype::TypeCommon;

    use crate::{
        iterator_traits::{IterGetSetSimd, ParStridedIteratorSimdZip, ShapeManipulator},
        par_strided_mut::par_strided_map_mut_simd::ParStridedMutSimd,
        with_simd::WithSimd,
    };
    /// A parallel SIMD-optimized map iterator over tensor elements.
    ///
    /// This struct allows for applying two separate functions (`f` and `f2`) to elements of a tensor
    /// in a SIMD-optimized and parallel manner.
    #[derive(Clone)]
    pub struct ParStridedMapSimd<'a, I, T: 'a, F, F2>
    where
        I: UnindexedProducer<Item = T>
            + 'a
            + IterGetSetSimd<Item = T>
            + ParallelIterator
            + ShapeManipulator,
    {
        /// The underlying parallel SIMD-optimized strided iterator.
        pub(crate) iter: I,
        /// The first function to apply to each item.
        pub(crate) f: F,
        /// The second function to apply to SIMD items.
        pub(crate) f2: F2,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<
            'a,
            I: UnindexedProducer<Item = T>
                + 'a
                + IterGetSetSimd<Item = T>
                + ParallelIterator
                + ShapeManipulator,
            T: 'a,
            F,
            F2,
        > ParStridedMapSimd<'a, I, T, F, F2>
    {
        /// Collects the results of the map operation into a new tensor.
        ///
        /// This method applies the provided functions `f` and `f2` to each element and SIMD item
        /// respectively, accumulating the results into a new tensor of type `U`.
        ///
        /// # Type Parameters
        ///
        /// * `U` - The type of the tensor to collect the results into. Must implement `TensorAlloc`, `Clone`,
        ///         and `TensorInfo`.
        ///
        /// # Arguments
        ///
        /// * `self` - The `ParStridedMapSimd` iterator instance.
        ///
        /// # Returns
        ///
        /// A new tensor of type `U` containing the results of the map operation.
        pub fn collect<U>(self) -> U
        where
            F: Fn((&mut <U as TensorAlloc>::Meta, <I as IterGetSetSimd>::Item)) + Sync + Send + 'a,
            U: Clone + TensorInfo<U::Meta> + TensorAlloc,
            <I as IterGetSetSimd>::Item: Send,
            <U as TensorAlloc>::Meta: CommonBounds,
            F2: Send
                + Sync
                + Copy
                + Fn(
                    (
                        MutVec<'_, <<U as TensorAlloc>::Meta as TypeCommon>::Vec>,
                        <I as IterGetSetSimd>::SimdItem,
                    ),
                ),
        {
            let res = U::_empty(self.iter.shape().clone()).unwrap();
            let par_strided = ParStridedMutSimd::new(res.clone());
            let zip = par_strided.zip(self.iter);
            let with_simd = WithSimd {
                base: zip,
                vec_op: self.f2,
            };
            with_simd.for_each(|x| {
                (self.f)(x);
            });
            res
        }
    }
}

/// A parallel map iterator over tensor elements.
///
/// This struct allows for applying a single function (`f`) to elements of a tensor
#[derive(Clone)]
pub struct ParStridedMap<'a, I, T: 'a, F>
where
    I: UnindexedProducer<Item = T> + 'a + IterGetSet<Item = T> + ParallelIterator,
{
    /// The underlying parallel strided iterator.
    pub(crate) iter: I,
    /// The function to apply to each item.
    pub(crate) f: F,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<
        'a,
        I: UnindexedProducer<Item = T> + 'a + IterGetSet<Item = T> + ParallelIterator,
        T: 'a,
        F,
    > ParStridedMap<'a, I, T, F>
{
    /// Collects the results of the map operation into a new tensor.
    ///
    /// This method applies the provided function `f` to each element of the tensor,
    /// accumulating the results into a new tensor of type `U`.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The type of the tensor to collect the results into. Must implement `TensorAlloc`, `Clone`,
    ///         and `TensorInfo`.
    ///
    /// # Arguments
    ///
    /// * `self` - The `ParStridedMap` iterator instance.
    ///
    /// # Returns
    ///
    /// A new tensor of type `U` containing the results of the map operation.
    pub fn collect<U>(self) -> U
    where
        F: Fn((&mut U::Meta, T)) + Sync + Send,
        U: Clone + TensorInfo<U::Meta> + TensorAlloc,
        <I as IterGetSet>::Item: Send,
        <U as TensorAlloc>::Meta: CommonBounds,
    {
        let res = U::_empty(self.iter.shape().clone()).unwrap();
        let strided_mut = ParStridedMapMut::new(res.clone());
        let zip = strided_mut.zip(self.iter);
        zip.for_each(|(x, y)| {
            (self.f)((x, y));
        });
        res
    }
}

impl<'a, T: CommonBounds> ShapeManipulator for ParStridedMutSimd<'a, T> {
    fn reshape<S: Into<hpt_common::shape::shape::Shape>>(self, shape: S) -> Self {
        let shape: Shape = shape.into();
        let new_base = self.base.reshape(shape);
        ParStridedMutSimd {
            base: new_base,
            phantom: self.phantom,
        }
    }

    fn transpose<AXIS: Into<hpt_common::axis::axis::Axis>>(self, axes: AXIS) -> Self {
        let axes = axes.into();
        let new_base = self.base.transpose(axes);
        ParStridedMutSimd {
            base: new_base,
            phantom: self.phantom,
        }
    }

    fn expand<S: Into<hpt_common::shape::shape::Shape>>(self, shape: S) -> Self {
        let shape: Shape = shape.into();
        let new_base = self.base.expand(shape);
        ParStridedMutSimd {
            base: new_base,
            phantom: self.phantom,
        }
    }
}
