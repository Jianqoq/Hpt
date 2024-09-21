use std::sync::Arc;
use tensor_common::{shape::Shape, strides::Strides};

use crate::iterator_traits::{IterGetSet, StridedIterator, StridedIteratorMap, StridedIteratorZip};

/// A module for zipped strided simd iterator.
pub mod strided_zip_simd {
    use tensor_common::{shape::Shape, strides::Strides};

    use crate::iterator_traits::{IterGetSetSimd, StridedIteratorSimd, StridedSimdIteratorZip};
    use std::sync::Arc;

    /// A single thread SIMD-optimized zipped iterator combining two iterators over tensor elements.
    ///
    /// # Example
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::StridedIteratorSimd;
    /// use tensor_dyn::TensorIterator;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// a.iter_simd().zip(a.iter_simd()).for_each(
    ///     |(x, y)| {
    ///         println!("{} {}", x, y);
    ///     },
    ///     |(x, y)| {
    ///         println!("{:?} {:?}", x, y);
    ///     },
    /// );
    /// ```
    #[derive(Clone)]
    pub struct StridedZipSimd<'a, A: 'a, B: 'a> {
        /// The first iterator to be zipped.
        pub(crate) a: A,
        /// The second iterator to be zipped.
        pub(crate) b: B,
        /// Phantom data to associate the lifetime `'a` with the struct.
        pub(crate) phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a, A, B> IterGetSetSimd for StridedZipSimd<'a, A, B>
    where
        A: IterGetSetSimd,
        B: IterGetSetSimd,
    {
        type Item = (<A as IterGetSetSimd>::Item, <B as IterGetSetSimd>::Item);

        type SimdItem = (
            <A as IterGetSetSimd>::SimdItem,
            <B as IterGetSetSimd>::SimdItem,
        );

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

        fn set_prg(&mut self, prg: Vec<i64>) {
            self.a.set_prg(prg.clone());
            self.b.set_prg(prg);
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

        fn next_simd(&mut self) {
            todo!()
        }
        #[inline(always)]
        fn inner_loop_next(&mut self, index: usize) -> Self::Item {
            (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
        }
        fn inner_loop_next_simd(&self, index: usize) -> Self::SimdItem {
            (
                self.a.inner_loop_next_simd(index),
                self.b.inner_loop_next_simd(index),
            )
        }
        fn all_last_stride_one(&self) -> bool {
            self.a.all_last_stride_one() && self.b.all_last_stride_one()
        }
        fn lanes(&self) -> Option<usize> {
            match (self.a.lanes(), self.b.lanes()) {
                (Some(a), Some(b)) => {
                    if a == b {
                        Some(a)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        
        fn layout(&self) -> &tensor_common::layout::Layout {
            self.a.layout()
        }
    }

    impl<'a, A, B> StridedZipSimd<'a, A, B>
    where
        A: 'a + IterGetSetSimd,
        B: 'a + IterGetSetSimd,
        <A as IterGetSetSimd>::Item: Send,
        <B as IterGetSetSimd>::Item: Send,
    {
        /// Creates a new `StridedZipSimd` instance by zipping two SIMD-optimized iterators.
        ///
        /// # Arguments
        ///
        /// * `a` - The first iterator to zip.
        /// * `b` - The second iterator to zip.
        ///
        /// # Returns
        ///
        /// A new `StridedZipSimd` instance that combines both iterators for synchronized iteration.
        pub fn new(a: A, b: B) -> Self {
            StridedZipSimd {
                a,
                b,
                phantom: std::marker::PhantomData,
            }
        }
    }

    impl<'a, A, B> StridedIteratorSimd for StridedZipSimd<'a, A, B>
    where
        A: IterGetSetSimd,
        B: IterGetSetSimd,
    {
    }
    impl<'a, A, B> StridedSimdIteratorZip for StridedZipSimd<'a, A, B>
    where
        A: IterGetSetSimd,
        B: IterGetSetSimd,
    {
    }
}

/// A single thread `non` SIMD-optimized zipped iterator combining two iterators over tensor elements.
///
/// # Example
/// ```
/// use tensor_dyn::tensor::Tensor;
/// use tensor_dyn::StridedIterator;
/// use tensor_dyn::TensorIterator;
/// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
/// a.iter().zip(a.iter()).for_each(|(x, y)| {
///     println!("{} {}", x, y);
/// });
/// ```
#[derive(Clone)]
pub struct StridedZip<'a, A: 'a, B: 'a> {
    /// The first iterator to be zipped.
    pub(crate) a: A,
    /// The second iterator to be zipped.
    pub(crate) b: B,
    /// Phantom data to associate the lifetime `'a` with the struct.
    pub(crate) phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, A, B> IterGetSet for StridedZip<'a, A, B>
where
    A: IterGetSet,
    B: IterGetSet,
{
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

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.a.set_prg(prg.clone());
        self.b.set_prg(prg);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        panic!("single thread strided zip does not support intervals");
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        self.a.broadcast_set_strides(shape);
        self.b.broadcast_set_strides(shape);
    }

    fn next(&mut self) {
        self.a.next();
        self.b.next();
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        (self.a.inner_loop_next(index), self.b.inner_loop_next(index))
    }
    
    fn strides(&self) -> &Strides {
        self.a.strides()
    }
    
    fn shape(&self) -> &Shape {
        self.a.shape()
    }
    
    fn outer_loop_size(&self) -> usize {
        self.a.outer_loop_size()
    }
    
    fn inner_loop_size(&self) -> usize {
        self.a.inner_loop_size()
    }
    
    fn layout(&self) -> &tensor_common::layout::Layout {
        self.a.layout()
    }
}

impl<'a, A, B> StridedZip<'a, A, B>
where
    A: 'a + IterGetSet,
    B: 'a + IterGetSet,
    <A as IterGetSet>::Item: Send,
    <B as IterGetSet>::Item: Send,
{
    /// Creates a new `StridedZip` instance by zipping two iterators.
    ///
    /// # Arguments
    ///
    /// * `a` - The first iterator to zip.
    /// * `b` - The second iterator to zip.
    ///
    /// # Returns
    ///
    /// A new `StridedZip` instance that combines both iterators for synchronized iteration.
    pub fn new(a: A, b: B) -> Self {
        StridedZip {
            a,
            b,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, A, B> StridedIteratorZip for StridedZip<'a, A, B> {}
impl<'a, A, B> StridedIteratorMap for StridedZip<'a, A, B> {}
impl<'a, A, B> StridedIterator for StridedZip<'a, A, B>
where
    A: IterGetSet,
    B: IterGetSet,
{
}
