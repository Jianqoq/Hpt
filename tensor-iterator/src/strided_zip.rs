use std::{ panic::Location, sync::Arc };
use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape, strides::Strides };

use crate::iterator_traits::{ IterGetSet, ShapeManipulator, StridedIterator };

#[cfg(feature = "simd")]
pub mod strided_zip_simd {
    use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape, strides::Strides };

    use crate::iterator_traits::{ IterGetSetSimd, ShapeManipulator, StridedIteratorSimd };
    use std::{ panic::Location, sync::Arc };

    #[derive(Clone)]
    pub struct StridedZipSimd<'a, A: 'a, B: 'a> {
        pub(crate) a: A,
        pub(crate) b: B,
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, A, B> IterGetSetSimd
        for StridedZipSimd<'a, A, B>
        where A: IterGetSetSimd, B: IterGetSetSimd
    {
        type Item = (<A as IterGetSetSimd>::Item, <B as IterGetSetSimd>::Item);

        type SimdItem = (<A as IterGetSetSimd>::SimdItem, <B as IterGetSetSimd>::SimdItem);

        fn set_end_index(&mut self, _: usize) {
            panic!("single thread strided zip does not support set_intervals");
        }

        fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
            panic!("single thread strided zip does not support set_intervals");
        }

        fn set_strides(&mut self, last_stride: Strides) {
            self.a.set_strides(last_stride.clone());
            self.b.set_strides(last_stride);
        }

        fn set_shape(&mut self, shape: Shape) {
            self.a.set_shape(shape.clone());
            self.b.set_shape(shape);
        }

        fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
            panic!("single thread strided zip does not support intervals");
        }

        fn strides(&self) -> &Strides {
            self.a.strides()
        }

        fn shape(&self) -> &Shape {
            self.a.shape()
        }

        fn broadcast_set_strides(&mut self, shape: &Shape) {
            self.a.broadcast_set_strides(shape);
            self.b.broadcast_set_strides(shape);
        }

        fn outer_loop_size(&self) -> usize {
            self.a.outer_loop_size()
        }

        fn inner_loop_size(&self) -> usize {
            self.a.inner_loop_size()
        }

        fn next(&mut self) {
            self.a.next();
            self.b.next();
        }

        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
        }

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.a.set_prg(prg.clone());
            self.b.set_prg(prg);
        }
        fn all_last_stride_one(&self) -> bool {
            self.a.all_last_stride_one() && self.b.all_last_stride_one()
        }
        fn lanes(&self) -> Option<usize> {
            match (self.a.lanes(), self.b.lanes()) {
                (Some(a), Some(b)) => {
                    if a == b { Some(a) } else { None }
                }
                _ => None,
            }
        }
        fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem {
            (self.a.inner_loop_next_simd(index), self.b.inner_loop_next_simd(index))
        }
        fn next_simd(&mut self) {
            todo!()
        }
    }

    impl<'a, A, B> StridedZipSimd<'a, A, B>
        where
            A: 'a + IterGetSetSimd,
            B: 'a + IterGetSetSimd,
            <A as IterGetSetSimd>::Item: Send,
            <B as IterGetSetSimd>::Item: Send
    {
        pub fn new(a: A, b: B) -> Self {
            StridedZipSimd {
                a,
                b,
                phantom: std::marker::PhantomData,
            }
        }

        #[track_caller]
        pub fn zip<C>(self, other: C) -> StridedZipSimd<'a, Self, C>
            where
                C: IterGetSetSimd + ShapeManipulator,
                Self: IterGetSetSimd + ShapeManipulator,
                <C as IterGetSetSimd>::Item: Send,
                <Self as IterGetSetSimd>::Item: Send
        {
            let new_shape = predict_broadcast_shape(
                &self.shape(),
                &other.shape(),
                Location::caller()
            ).expect("Cannot broadcast shapes");

            let mut a = self.reshape(new_shape.clone());
            let mut b = other.reshape(new_shape.clone());

            a.set_shape(new_shape.clone());
            b.set_shape(new_shape.clone());
            StridedZipSimd::new(a, b)
        }
    }

    impl<'a, A, B> StridedIteratorSimd
        for StridedZipSimd<'a, A, B>
        where A: IterGetSetSimd, B: IterGetSetSimd
    {
        type Item = <Self as IterGetSetSimd>::Item;
        type SimdItem = <Self as IterGetSetSimd>::SimdItem;

        fn for_each<F, F2>(mut self, folder: F, folder2: F2)
            where F: Fn(Self::Item), F2: Fn(Self::SimdItem)
        {
            let outer_loop_size = self.outer_loop_size();
            let inner_loop_size = self.inner_loop_size(); // we don't need to add 1 as we didn't subtract shape by 1
            self.set_prg(vec![0; self.a.shape().len()]);
            match (self.all_last_stride_one(), self.lanes()) {
                (true, Some(vec_size)) => {
                    let remain = inner_loop_size % vec_size;
                    let inner = inner_loop_size - remain;
                    let n = inner / vec_size;
                    let unroll = n % 4;
                    if remain > 0 {
                        if unroll == 0 {
                            for _ in 0..outer_loop_size {
                                for idx in 0..n / 4 {
                                    folder2(self.inner_loop_next_simd(idx * 4));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 1));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 2));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 3));
                                }
                                for idx in inner..inner_loop_size {
                                    folder(self.inner_loop_next(idx));
                                }
                                self.next();
                            }
                        } else {
                            for _ in 0..outer_loop_size {
                                for idx in 0..n {
                                    folder2(self.inner_loop_next_simd(idx));
                                }
                                for idx in inner..inner_loop_size {
                                    folder(self.inner_loop_next(idx));
                                }
                                self.next();
                            }
                        }
                    } else {
                        if unroll == 0 {
                            for _ in 0..outer_loop_size {
                                for idx in 0..n / 4 {
                                    folder2(self.inner_loop_next_simd(idx * 4));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 1));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 2));
                                    folder2(self.inner_loop_next_simd(idx * 4 + 3));
                                }
                                self.next();
                            }
                        } else {
                            for _ in 0..outer_loop_size {
                                for idx in 0..n {
                                    folder2(self.inner_loop_next_simd(idx));
                                }
                                self.next();
                            }
                        }
                    }
                }
                _ => {
                    for _ in 0..outer_loop_size {
                        for idx in 0..inner_loop_size {
                            folder(self.inner_loop_next(idx));
                        }
                        self.next();
                    }
                }
            }
        }

        fn for_each_init<F, INIT, T>(mut self, init: INIT, func: F)
            where F: Fn(&mut T, Self::Item), INIT: Fn() -> T
        {
            let outer_loop_size = self.outer_loop_size();
            let inner_loop_size = self.inner_loop_size();
            let mut init = init();
            for _ in 0..outer_loop_size {
                for idx in 0..inner_loop_size {
                    func(&mut init, self.inner_loop_next(idx));
                }
                self.next();
            }
        }
    }
}

#[derive(Clone)]
pub struct StridedZip<'a, A: 'a, B: 'a> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, A, B> IterGetSet for StridedZip<'a, A, B> where A: IterGetSet, B: IterGetSet {
    type Item = (<A as IterGetSet>::Item, <B as IterGetSet>::Item);

    fn set_end_index(&mut self, _: usize) {
        panic!("single thread strided zip does not support set_intervals");
    }

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
        panic!("single thread strided zip does not support set_intervals");
    }

    fn set_strides(&mut self, last_stride: Strides) {
        self.a.set_strides(last_stride.clone());
        self.b.set_strides(last_stride);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.a.set_shape(shape.clone());
        self.b.set_shape(shape);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        panic!("single thread strided zip does not support intervals");
    }

    fn strides(&self) -> &Strides {
        self.a.strides()
    }

    fn shape(&self) -> &Shape {
        self.a.shape()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        self.a.broadcast_set_strides(shape);
        self.b.broadcast_set_strides(shape);
    }

    fn outer_loop_size(&self) -> usize {
        self.a.outer_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.a.inner_loop_size()
    }

    fn next(&mut self) {
        self.a.next();
        self.b.next();
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.a.set_prg(prg.clone());
        self.b.set_prg(prg);
    }
}

impl<'a, A, B> StridedZip<'a, A, B>
    where
        A: 'a + IterGetSet,
        B: 'a + IterGetSet,
        <A as IterGetSet>::Item: Send,
        <B as IterGetSet>::Item: Send
{
    pub fn new(a: A, b: B) -> Self {
        StridedZip {
            a,
            b,
            phantom: std::marker::PhantomData,
        }
    }

    #[track_caller]
    pub fn zip<C>(self, other: C) -> StridedZip<'a, Self, C>
        where
            C: IterGetSet + ShapeManipulator,
            Self: IterGetSet + ShapeManipulator,
            <C as IterGetSet>::Item: Send,
            <Self as IterGetSet>::Item: Send
    {
        let new_shape = predict_broadcast_shape(
            &self.shape(),
            &other.shape(),
            Location::caller()
        ).expect("Cannot broadcast shapes");

        let mut a = self.reshape(new_shape.clone());
        let mut b = other.reshape(new_shape.clone());

        a.set_shape(new_shape.clone());
        b.set_shape(new_shape.clone());
        StridedZip::new(a, b)
    }
}

impl<'a, A, B> StridedIterator for StridedZip<'a, A, B> where A: IterGetSet, B: IterGetSet {
    type Item = <Self as IterGetSet>::Item;

    fn for_each<F>(mut self, folder: F) where F: Fn(Self::Item) {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size(); // we don't need to add 1 as we didn't subtract shape by 1
        self.set_prg(vec![0; self.a.shape().len()]);
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                folder(self.inner_loop_next(idx));
            }
            self.next();
        }
    }

    fn for_each_init<F, INIT, T>(mut self, init: INIT, func: F)
        where F: Fn(&mut T, Self::Item), INIT: Fn() -> T
    {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size();
        let mut init = init();
        for _ in 0..outer_loop_size {
            for idx in 0..inner_loop_size {
                func(&mut init, self.inner_loop_next(idx));
            }
            self.next();
        }
    }
}
