use std::sync::Arc;
use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use crate::{
    iterator_traits::{ IterGetSet, StridedIterator },
    strided::Strided,
    strided_zip::StridedZip,
};

#[cfg(feature = "simd")]
pub mod simd_imports {
    use std::sync::Arc;
    use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape };
    use tensor_traits::{ CommonBounds, TensorInfo };
    use tensor_types::dtype::TypeCommon;
    use tensor_types::vectors::traits::VecCommon;
    use crate::{
        iterator_traits::{ IterGetSetSimd, StridedIteratorSimd },
        strided::strided_simd::StridedSimd,
        strided_zip::strided_zip_simd::StridedZipSimd,
    };

    pub struct StridedMutSimd<'a, T: TypeCommon> {
        pub(crate) base: StridedSimd<T>,
        pub(crate) last_stride: i64,
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, T: CommonBounds> StridedMutSimd<'a, T> {
        pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
            let base = StridedSimd::new(tensor);
            let last_stride = base.last_stride;
            StridedMutSimd {
                base,
                last_stride,
                phantom: std::marker::PhantomData,
            }
        }

        #[track_caller]
        pub fn zip<C>(mut self, mut other: C) -> StridedZipSimd<'a, Self, C>
            where C: 'a + IterGetSetSimd, <C as IterGetSetSimd>::Item: Send
        {
            let new_shape = predict_broadcast_shape(self.shape(), other.shape()).expect(
                "Cannot broadcast shapes"
            );

            other.broadcast_set_strides(&new_shape);
            self.broadcast_set_strides(&new_shape);

            other.set_shape(new_shape.clone());
            self.set_shape(new_shape.clone());

            StridedZipSimd::new(self, other)
        }
    }

    #[cfg(feature = "simd")]
    impl<'a, T> StridedIteratorSimd for StridedMutSimd<'a, T> where T: CommonBounds {
        type Item = &'a mut T;
        type SimdItem = &'a mut <T as TypeCommon>::Vec;

        fn for_each<F, F2>(mut self, func: F, _: F2) where F: Fn(Self::Item), F2: Fn(Self::SimdItem) {
            let outer_loop_size = self.outer_loop_size();
            let inner_loop_size = self.inner_loop_size() + 1;
            for _ in 0..outer_loop_size {
                for idx in 0..inner_loop_size {
                    func(self.inner_loop_next(idx));
                }
                self.next();
            }
        }

        fn for_each_init<F, INIT, I>(mut self, init: INIT, func: F)
            where F: Fn(&mut I, Self::Item), INIT: Fn() -> I
        {
            let outer_loop_size = self.outer_loop_size();
            let inner_loop_size = self.inner_loop_size() + 1;
            let mut init = init();
            for _ in 0..outer_loop_size {
                for idx in 0..inner_loop_size {
                    func(&mut init, self.inner_loop_next(idx));
                }
                self.next();
            }
        }
    }

    impl<'a, T: 'a> IterGetSetSimd for StridedMutSimd<'a, T> where T: CommonBounds {
        type Item = &'a mut T;
        type SimdItem = &'a mut <T as TypeCommon>::Vec;

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

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.base.set_prg(prg);
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

        fn next_simd(&mut self) {
            todo!()
        }
        #[inline(always)]
        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            unsafe {
                &mut *self.base.ptr.ptr.offset((index as isize) * (self.last_stride as isize))
            }
        }
        fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem {
            let vector = unsafe { self.base.ptr.ptr.add(index * T::Vec::SIZE) };
            unsafe { std::mem::transmute(vector) }
        }
        fn all_last_stride_one(&self) -> bool {
            self.base.all_last_stride_one()
        }
        fn lanes(&self) -> Option<usize> {
            self.base.lanes()
        }
    }
}

pub struct StridedMut<'a, T> {
    pub(crate) base: Strided<T>,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: CommonBounds> StridedMut<'a, T> {
    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        StridedMut {
            base: Strided::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    #[track_caller]
    pub fn zip<C>(mut self, mut other: C) -> StridedZip<'a, Self, C>
        where C: 'a + IterGetSet, <C as IterGetSet>::Item: Send
    {
        let new_shape = match predict_broadcast_shape(self.shape(), other.shape()) {
            Ok(s) => s,
            Err(err) => {
                panic!("{}", err);
            }
        };

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        StridedZip::new(self, other)
    }
}

impl<'a, T> StridedIterator for StridedMut<'a, T> where T: CommonBounds {
    type Item = &'a mut T;

    fn for_each<F>(mut self, func: F) where F: Fn(Self::Item) {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(self.inner_loop_next(idx));
            }
            self.next();
        }
    }

    fn for_each_init<F, INIT, I>(mut self, init: INIT, func: F)
        where F: Fn(&mut I, Self::Item), INIT: Fn() -> I
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size() + 1;
        let mut init = init();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(&mut init, self.inner_loop_next(idx));
            }
            self.next();
        }
    }
}

impl<'a, T: 'a> IterGetSet for StridedMut<'a, T> where T: CommonBounds {
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

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.base.set_prg(prg);
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
}
