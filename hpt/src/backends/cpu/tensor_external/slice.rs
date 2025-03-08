use std::{cell::RefCell, rc::Rc};

use crate::tensor::{DiffTensor, Tensor};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::error::base::TensorError;
use hpt_iterator::{iterator_traits::ParStridedIteratorZip, TensorIterator};
use hpt_traits::{
    ops::{creation::TensorCreator, slice::Slice},
    tensor::CommonBounds,
};
use num::Integer;
use rayon::iter::ParallelIterator;
impl<T, const DEVICE: usize, Al> Slice for DiffTensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    fn slice(
        &self,
        index: &[(i64, i64, i64)],
    ) -> Result<DiffTensor<T, Cpu, DEVICE, Al>, TensorError> {
        let res = self.inner.slice(index)?;
        let lhs = self.clone();
        if let None = lhs.grad.borrow().as_ref() {
            lhs.grad.replace(Some(self.inner.zeros_like()?));
        }
        lhs.out_degree.borrow_mut().inc();
        let index = index.to_vec();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE, Al>| {
                let taked = lhs.grad.take();
                if let Some(tmp) = taked {
                    let mut sliced = tmp.slice(&index)?;
                    sliced
                        .par_iter_mut()
                        .zip(grad.inner.par_iter())
                        .for_each(|(a, b)| {
                            *a = a._add(b);
                        });
                    if *lhs.out_degree.borrow() > 1 {
                        *lhs.out_degree.borrow_mut() -= 1;
                    } else {
                        lhs.backward.borrow_mut()(tmp)?;
                    }
                } else {
                    panic!("Gradient is not set for slice");
                }
                Ok(false)
            })),
        })
    }
}
