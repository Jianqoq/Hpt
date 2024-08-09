use std::sync::Arc;
use tensor_common::{ shape::Shape, shape_utils::predict_broadcast_shape, strides::Strides };

use crate::iterator_traits::{ IterGetSet, ShapeManipulator, StridedIterator };

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

    pub fn zip<C>(self, other: C) -> StridedZip<'a, Self, C>
        where
            C: IterGetSet + ShapeManipulator,
            Self: IterGetSet + ShapeManipulator,
            <C as IterGetSet>::Item: Send,
            <Self as IterGetSet>::Item: Send
    {
        let new_shape = predict_broadcast_shape(&self.shape(), &other.shape()).expect(
            "Cannot broadcast shapes"
        );

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
}
