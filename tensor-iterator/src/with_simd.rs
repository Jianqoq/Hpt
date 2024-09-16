use std::sync::Arc;

use rayon::iter::{
    plumbing::{ bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer },
    ParallelIterator,
};
use tensor_common::{ shape::Shape, strides::Strides };

use crate::iterator_traits::IterGetSetSimd;

pub(crate) struct WithSimd<I, F> {
    pub(crate) base: I,
    pub(crate) vec_op: F,
}

impl<I, F> UnindexedProducer
    for WithSimd<I, F>
    where
        I: UnindexedProducer + IterGetSetSimd + ParallelIterator,
        F: Fn(<I as IterGetSetSimd>::SimdItem) + Sync + Send + Copy
{
    type Item = <I as IterGetSetSimd>::Item;
    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.base.split();
        (
            WithSimd {
                base: a,
                vec_op: self.vec_op,
            },
            b.map(|x| WithSimd {
                base: x,
                vec_op: self.vec_op,
            }),
        )
    }

    fn fold_with<FOLD>(mut self, mut folder: FOLD) -> FOLD where FOLD: Folder<Self::Item> {
        let outer_loop_size = self.outer_loop_size();
        let inner_loop_size = self.inner_loop_size() + 1;
        let vec_op = self.vec_op;
        println!("lanes: {:?}", self.lanes());
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
                                vec_op(self.inner_loop_next_simd(idx * 4));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 1));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 2));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 3));
                            }
                            for idx in inner..inner_loop_size {
                                folder = folder.consume(self.inner_loop_next(idx));
                            }
                            self.next();
                        }
                    } else {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n {
                                vec_op(self.inner_loop_next_simd(idx));
                            }
                            for idx in n * vec_size..inner_loop_size {
                                folder = folder.consume(self.inner_loop_next(idx));
                            }
                            self.next();
                        }
                    }
                } else {
                    if unroll == 0 {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n / 4 {
                                vec_op(self.inner_loop_next_simd(idx * 4));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 1));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 2));
                                vec_op(self.inner_loop_next_simd(idx * 4 + 3));
                            }
                            self.next();
                        }
                    } else {
                        for _ in 0..outer_loop_size {
                            for idx in 0..n {
                                vec_op(self.inner_loop_next_simd(idx));
                            }
                            self.next();
                        }
                    }
                }
            }
            _ => {
                for _ in 0..outer_loop_size {
                    for idx in 0..inner_loop_size {
                        folder = folder.consume(self.inner_loop_next(idx));
                    }
                    self.next();
                }
            }
        }
        folder
    }
}

impl<I, F> ParallelIterator
    for WithSimd<I, F>
    where
        I: UnindexedProducer + IterGetSetSimd + ParallelIterator,
        F: Fn(<I as IterGetSetSimd>::SimdItem) + Sync + Send + Copy,
        <I as IterGetSetSimd>::Item: Send
{
    type Item = <I as IterGetSetSimd>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where C: UnindexedConsumer<Self::Item> {
        bridge_unindexed(self, consumer)
    }
}

impl<I, F> IterGetSetSimd
    for WithSimd<I, F>
    where
        I: UnindexedProducer + IterGetSetSimd + ParallelIterator,
        F: Fn(<I as IterGetSetSimd>::SimdItem) + Sync + Send + Copy
{
    type Item = <I as IterGetSetSimd>::Item;

    type SimdItem = <I as IterGetSetSimd>::SimdItem;

    fn set_end_index(&mut self, _: usize) {
        panic!("single thread strided zip does not support set_intervals");
    }

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
        panic!("single thread strided zip does not support set_intervals");
    }

    fn set_strides(&mut self, last_stride: Strides) {
        self.base.set_strides(last_stride.clone());
    }

    fn set_shape(&mut self, shape: Shape) {
        self.base.set_shape(shape);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        panic!("single thread strided zip does not support intervals");
    }

    fn strides(&self) -> &Strides {
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
        self.base.inner_loop_next(index)
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
    fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem {
        self.base.inner_loop_next_simd(index)
    }
    fn next_simd(&mut self) {
        todo!()
    }
}
