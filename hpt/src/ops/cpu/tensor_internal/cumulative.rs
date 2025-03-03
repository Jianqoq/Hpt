use crate::ops::cpu::utils::unary::unary::cumulate;
use crate::tensor_base::_Tensor;
use crate::Cpu;
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_common::error::base::TensorError;
use hpt_traits::CommonBounds;
use hpt_traits::CumulativeOps;

impl<T: CommonBounds, const DEVICE: usize, A2> CumulativeOps for _Tensor<T, Cpu, DEVICE, A2>
where
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    fn cumsum<A: Into<Option<i64>>>(&self, axis: A) -> std::result::Result<Self, TensorError> {
        cumulate(self, axis, T::ZERO, |a, b| a._add(b))
    }

    fn cumprod<A: Into<Option<i64>>>(&self, axis: A) -> std::result::Result<Self, TensorError> {
        cumulate(self, axis, T::ONE, |a, b| a._mul(b))
    }
}
