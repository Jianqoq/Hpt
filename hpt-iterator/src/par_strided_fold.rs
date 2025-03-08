use std::fmt::Display;

use crate::CommonBounds;
use rayon::iter::{
    plumbing::{bridge_unindexed, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};

use crate::iterator_traits::IterGetSet;

/// A parallel strided fold iterator over tensor elements.
///
/// This struct facilitates performing fold (reduction) operations on tensor elements in a parallel and strided manner.
/// It leverages Rayon for concurrent execution, ensuring efficient traversal and aggregation of tensor data based on
/// their strides.
pub struct ParStridedFold<I, ID, F> {
    pub(crate) iter: I,
    pub(crate) identity: ID,
    pub(crate) fold_op: F,
}

impl<I, ID, F> ParallelIterator for ParStridedFold<I, ID, F>
where
    I: ParallelIterator + UnindexedProducer + IterGetSet,
    F: Fn(ID, <I as IterGetSet>::Item) -> ID + Sync + Send + Copy,
    ID: CommonBounds,
    <I as IterGetSet>::Item: Display,
{
    type Item = ID;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

impl<I, ID, F> UnindexedProducer for ParStridedFold<I, ID, F>
where
    I: ParallelIterator + UnindexedProducer + IterGetSet,
    F: Fn(ID, <I as IterGetSet>::Item) -> ID + Sync + Send + Copy,
    ID: CommonBounds,
    <I as IterGetSet>::Item: Display,
{
    type Item = ID;

    fn split(self) -> (Self, Option<Self>) {
        let (a, b) = self.iter.split();
        (
            ParStridedFold {
                iter: a,
                identity: self.identity,
                fold_op: self.fold_op,
            },
            b.map(|b| ParStridedFold {
                iter: b,
                identity: self.identity,
                fold_op: self.fold_op,
            }),
        )
    }

    fn fold_with<FD>(mut self, mut folder: FD) -> FD
    where
        FD: rayon::iter::plumbing::Folder<Self::Item>,
    {
        let init = self.identity;
        let outer_loop_size = self.iter.outer_loop_size();
        let inner_loop_size = self.iter.inner_loop_size() + 1; // parallel iterator will auto subtract 1
        for _ in 0..outer_loop_size {
            for i in 0..inner_loop_size {
                let item = self.iter.inner_loop_next(i);
                let val = (self.fold_op)(init, item);
                folder = folder.consume(val);
            }
            self.iter.next();
        }
        folder
    }
}
