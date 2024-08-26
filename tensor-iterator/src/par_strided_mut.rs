use std::{ panic::Location, sync::Arc };
use rayon::iter::{
    plumbing::{ bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer },
    ParallelIterator,
};
use tensor_common::{ shape::Shape, shape_utils::{ mt_intervals, predict_broadcast_shape } };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use tensor_types::dtype::TypeCommon;
use crate::{ iterator_traits::IterGetSet, par_strided::ParStrided, par_strided_zip::ParStridedZip };

#[cfg(feature = "simd")]
pub mod par_strided_map_mut_simd {
    use std::{panic::Location, sync::Arc};
    use tensor_types::vectors::traits::{Init, VecTrait};
    use rayon::iter::{plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer}, ParallelIterator};
    use tensor_common::{shape::Shape, shape_utils::{mt_intervals, predict_broadcast_shape}};
    use tensor_traits::{CommonBounds, TensorInfo};
    use tensor_types::dtype::TypeCommon;
    use tensor_types::vectors::traits::VecSize;
    use crate::{iterator_traits::IterGetSetSimd, par_strided::par_strided_simd::ParStridedSimd, par_strided_zip::par_strided_zip_simd::ParStridedZipSimd};

    pub struct ParStridedMutSimd<'a, T: TypeCommon> {
        pub(crate) base: ParStridedSimd<T>,
        pub(crate) vector: T::Vec,
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, T: CommonBounds> ParStridedMutSimd<'a, T> {
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            ParStridedMutSimd {
                base: ParStridedSimd::new(tensor),
                vector: T::Vec::splat(T::ZERO),
                phantom: std::marker::PhantomData,
            }
        }
    
        #[track_caller]
        pub fn zip<C>(mut self, mut other: C) -> ParStridedZipSimd<'a, Self, C>
            where
                C: UnindexedProducer + 'a + IterGetSetSimd + ParallelIterator,
                <C as IterGetSetSimd>::Item: Send,
                <T as TypeCommon>::Vec: Send
        {
            let new_shape = predict_broadcast_shape(
                self.shape(),
                other.shape(),
                Location::caller()
            ).expect("Cannot broadcast shapes");
    
            let inner_loop_size = new_shape[new_shape.len() - 1] as usize;
    
            // if collapse all is true, then the outer loop size is the product of all the elements in the shape
            // inner_loop_size in this case will be useless
            let outer_loop_size = (new_shape.size() as usize) / inner_loop_size;
            let num_threads;
            if outer_loop_size < rayon::current_num_threads() {
                num_threads = outer_loop_size;
            } else {
                num_threads = rayon::current_num_threads();
            }
            let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
            let len = intervals.len();
            self.set_intervals(intervals.clone());
            self.set_end_index(len);
            other.set_intervals(intervals.clone());
            other.set_end_index(len);
    
            other.broadcast_set_strides(&new_shape);
            self.broadcast_set_strides(&new_shape);
    
            other.set_shape(new_shape.clone());
            self.set_shape(new_shape.clone());
    
            ParStridedZipSimd::new(self, other)
        }
    }

    impl<'a, T> ParallelIterator
        for ParStridedMutSimd<'a, T>
        where T: CommonBounds, <T as TypeCommon>::Vec: Send
    {
        type Item = &'a mut T;
    
        fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
            bridge_unindexed(self, consumer)
        }
    }

    impl<'a, T> UnindexedProducer
        for ParStridedMutSimd<'a, T>
        where T: CommonBounds, <T as TypeCommon>::Vec: Send
    {
        type Item = &'a mut T;
    
        fn split(self) -> (Self, Option<Self>) {
            let (a, b) = self.base.split();
            (
                ParStridedMutSimd {
                    base: a,
                    phantom: std::marker::PhantomData,
                    vector: T::Vec::splat(T::ZERO),
                },
                b.map(|x| ParStridedMutSimd {
                    base: x,
                    phantom: std::marker::PhantomData,
                    vector: T::Vec::splat(T::ZERO),
                }),
            )
        }
    
        fn fold_with<F>(self, folder: F) -> F where F: Folder<Self::Item> {
            folder
        }
    }

    impl<'a, T: 'a> IterGetSetSimd for ParStridedMutSimd<'a, T> where T: CommonBounds, <T as TypeCommon>::Vec: Send {
        type Item = &'a mut T;
    
        type SimdItem = &'a mut <T as TypeCommon>::Vec where Self: 'a;
    
        fn set_end_index(&mut self, end_index: usize) {
            self.base.set_end_index(end_index);
        }
    
        fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
            self.base.set_intervals(intervals);
        }
    
        fn set_strides(&mut self, strides: tensor_common::strides::Strides) {
            self.base.set_strides(strides);
        }
    
        fn set_shape(&mut self, shape: Shape) {
            self.base.set_shape(shape);
        }
    
        fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
            self.base.intervals()
        }
    
        fn strides(&self) -> &tensor_common::strides::Strides {
            self.base.strides()
        }
    
        fn shape(&self) -> &Shape {
            self.base.shape()
        }
    
        fn broadcast_set_strides(&mut self, shape: &Shape) {
            self.base.broadcast_set_strides(shape);
        }
    
        fn outer_loop_size(&self) -> usize {
            self.base.outer_loop_size()
        }
    
        fn inner_loop_size(&self) -> usize {
            self.base.inner_loop_size()
        }
    
        fn next(&mut self) {
            self.base.next();
        }
    
        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            unsafe {
                self.base.ptr
                    .get_ptr()
                    .add(index * (self.base.last_stride as usize))
                    .as_mut()
                    .unwrap()
            }
        }
    
        fn set_prg(&mut self, prg: Vec<i64>) {
            self.base.set_prg(prg);
        }
    
        fn all_last_stride_one(&self) -> bool {
            self.base.all_last_stride_one()
        }
    
        fn lanes(&self) -> Option<usize> {
            self.base.lanes()
        }
    
        fn inner_loop_next_simd(& self, index: usize) -> Self::SimdItem {
            unsafe { 
                std::ptr::copy_nonoverlapping(self.base.ptr.get_ptr().add(index * T::Vec::SIZE), self.vector.as_mut_ptr_uncheck(), T::Vec::SIZE);
                std::mem::transmute(self.vector.as_mut_ptr_uncheck().as_mut().unwrap())
            }
        }
    
        fn next_simd(&mut self) {
            self.base.next_simd();
        }
    }
    
}

pub struct ParStridedMut<'a, T> {
    pub(crate) base: ParStrided<T>,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: CommonBounds> ParStridedMut<'a, T> {
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        ParStridedMut {
            base: ParStrided::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> ParStridedZip<'a, Self, C>
        where
            C: UnindexedProducer + 'a + IterGetSet + ParallelIterator,
            <C as IterGetSet>::Item: Send,
            <T as TypeCommon>::Vec: Send
    {
        let new_shape = predict_broadcast_shape(
            self.shape(),
            other.shape(),
            Location::caller()
        ).expect("Cannot broadcast shapes");

        let inner_loop_size = new_shape[new_shape.len() - 1] as usize;

        // if collapse all is true, then the outer loop size is the product of all the elements in the shape
        // inner_loop_size in this case will be useless
        let outer_loop_size = (new_shape.size() as usize) / inner_loop_size;
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
        let len = intervals.len();
        self.set_intervals(intervals.clone());
        self.set_end_index(len);
        other.set_intervals(intervals.clone());
        other.set_end_index(len);

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        ParStridedZip::new(self, other)
    }
}

impl<'a, T> ParallelIterator
    for ParStridedMut<'a, T>
    where T: CommonBounds
{
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
        bridge_unindexed(self, consumer)
    }
}

impl<'a, T> UnindexedProducer
    for ParStridedMut<'a, T>
    where T: CommonBounds
{
    type Item = &'a mut T;

    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.base.split();
        (
            ParStridedMut {
                base: a,
                phantom: std::marker::PhantomData,
            },
            b.map(|x| ParStridedMut {
                base: x,
                phantom: std::marker::PhantomData,
            }),
        )
    }

    fn fold_with<F>(self, folder: F) -> F where F: Folder<Self::Item> {
        folder
    }
}

impl<'a, T: 'a> IterGetSet for ParStridedMut<'a, T> where T: CommonBounds {
    type Item = &'a mut T;

    fn set_end_index(&mut self, end_index: usize) {
        self.base.set_end_index(end_index);
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.base.set_intervals(intervals);
    }

    fn set_strides(&mut self, strides: tensor_common::strides::Strides) {
        self.base.set_strides(strides);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.base.set_shape(shape);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        self.base.intervals()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.base.strides()
    }

    fn shape(&self) -> &Shape {
        self.base.shape()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        self.base.broadcast_set_strides(shape);
    }

    fn outer_loop_size(&self) -> usize {
        self.base.outer_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.base.inner_loop_size()
    }

    fn next(&mut self) {
        self.base.next();
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        unsafe {
            self.base.ptr
                .get_ptr()
                .add(index * (self.base.last_stride as usize))
                .as_mut()
                .unwrap()
        }
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.base.set_prg(prg);
    }
}

