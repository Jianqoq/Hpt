use crate::iterator_traits::StridedIterator;
use crate::{iterator_traits::IterGetSet, strided_map_mut::StridedMapMut};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};

/// A module for strided map simd iterator.
pub mod strided_map_simd {
    use crate::iterator_traits::StridedIteratorSimd;
    use crate::{
        iterator_traits::IterGetSetSimd, strided_map_mut::strided_map_mut_simd::StridedMapMutSimd,
    };
    use crate::{CommonBounds, TensorInfo};
    use hpt_traits::ops::creation::TensorCreator;
    use hpt_types::dtype::TypeCommon;

    /// # StridedMapSimd
    /// A structure representing a SIMD (Single Instruction, Multiple Data) version of the strided map operation.
    ///
    /// This structure allows applying SIMD operations on tensors or arrays with strided memory layouts,
    /// optimizing performance for certain types of computations.
    #[derive(Clone)]
    pub struct StridedMapSimd<'a, I, T: 'a, F, F2>
    where
        I: 'a + IterGetSetSimd<Item = T>,
    {
        /// An iterator implementing the `IterGetSetSimd` trait, providing access to the tensor elements.
        pub(crate) iter: I,
        /// A function to be applied to individual elements of the tensor
        pub(crate) f: F,
        /// A function to be applied to SIMD items of the tensor
        pub(crate) f2: F2,
        /// A marker for lifetimes to ensure proper memory safety
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, I: 'a + IterGetSetSimd<Item = T>, T: 'a, F, F2> StridedMapSimd<'a, I, T, F, F2> {
        /// Collects the results of applying the SIMD operations into a new tensor of type `U`.
        ///
        /// This method applies two functions, `f` for individual elements and `f2` for SIMD elements,
        /// and collects the results into a new tensor of type `U`.
        ///
        /// # Type Parameters
        /// - `U`: The type of the resulting tensor, which must implement `TensorAlloc` and `TensorInfo`.
        ///
        /// # Returns
        /// A new tensor of type `U` containing the results of applying the functions.
        pub fn collect<U>(self) -> U
        where
            F: Fn(T) -> U::Meta + Sync + Send + 'a,
            F2: Fn(
                    <I as IterGetSetSimd>::SimdItem,
                ) -> <<U as TensorCreator>::Meta as TypeCommon>::Vec
                + Sync
                + Send
                + 'a,
            U: Clone + TensorInfo + TensorCreator<Output = U>,
            <I as IterGetSetSimd>::Item: Send,
            <U as TensorCreator>::Meta: CommonBounds,
            <<U as TensorCreator>::Meta as TypeCommon>::Vec: Send,
        {
            let res = U::empty(self.iter.shape().clone()).unwrap();
            let strided_mut = StridedMapMutSimd::new(res.clone());
            let zip = strided_mut.zip(self.iter);
            zip.for_each(
                |(x, y)| {
                    *x = (self.f)(y);
                },
                |(x, y)| {
                    *x = (self.f2)(y);
                },
            );
            res
        }
    }
}

/// # StridedMap
///
/// This struct is used to apply a function over a tensor or array that has a strided memory layout.
/// The function `f` is applied to each element in the tensor as accessed through the strided iterator.
///
/// # Type Parameters
/// - `'a`: The lifetime associated with the data being processed.
/// - `I`: The iterator type that implements the `IterGetSet` trait, providing access to the elements.
/// - `T`: The type of the elements in the tensor.
/// - `F`: The function type that is applied to the tensor's elements.
#[derive(Clone)]
pub struct StridedMap<'a, I, T: 'a, F>
where
    I: 'a + IterGetSet<Item = T>,
{
    /// The iterator providing access to the tensor elements.
    pub(crate) iter: I,
    /// The function to be applied to each element of the tensor.
    pub(crate) f: F,
    /// Marker for the lifetime `'a`.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}
impl<'a, I: 'a + IterGetSet<Item = T>, T: 'a, F> StridedMap<'a, I, T, F> {
    /// Collects the results of applying the function `f` to the elements of the tensor into a new tensor of type `U`.
    ///
    /// This method applies the function `f` to each element in the tensor and collects the results in a new tensor `U`.
    ///
    /// # Type Parameters
    /// - `U`: The type of the resulting tensor, which must implement the `TensorAlloc` and `TensorInfo` traits.
    ///
    /// # Returns
    /// Returns a new tensor of type `U` containing the results of applying the function `f`.
    pub fn collect<U>(self) -> U
    where
        F: Fn(T) -> U::Meta + Sync + Send + 'a,
        U: Clone + TensorInfo + TensorCreator<Output = U>,
        <I as IterGetSet>::Item: Send,
        <U as TensorCreator>::Meta: CommonBounds,
    {
        let res = U::empty(self.iter.shape().clone()).unwrap();
        let strided_mut = StridedMapMut::new(res.clone());
        let zip = strided_mut.zip(self.iter);
        zip.for_each(|(x, y)| {
            *x = (self.f)(y);
        });
        res
    }
}
