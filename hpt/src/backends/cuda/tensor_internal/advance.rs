use cudarc::driver::DeviceRepr;
use hpt_common::error::base::TensorError;
use hpt_cudakernels::PAD;
use hpt_traits::{
    ops::{
        advance::{AdvancedOps, HardMax, TensorWhere},
        creation::TensorCreator,
    },
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::{
    dtype::CudaType,
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};

use crate::{
    backend::Cuda,
    backends::cuda::cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
    tensor_base::_Tensor,
};
use cudarc::driver::LaunchAsync;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
impl<T: CommonBounds + PartialOrd + DeviceRepr + CudaType, const DEVICE: usize, Al> AdvancedOps
    for _Tensor<T, Cuda, DEVICE, Al>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = _Tensor<T, Cuda, DEVICE, Al>;

    type IndexOutput = _Tensor<i64, Cuda, DEVICE, Al>;

    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError> {
        let res_shape = self
            .shape()
            .iter()
            .zip(pads.iter())
            .map(|(x, (a, b))| x + a + b)
            .collect::<Vec<_>>();

        let res = _Tensor::<T, Cuda, DEVICE, Al>::full(val, &res_shape)?;
        let mut pads = pads.to_vec();
        if pads.len() < self.ndim() {
            pads.resize(self.ndim(), (0, 0));
        }
        let pads_start = pads.iter().map(|(a, _)| *a).collect::<Vec<_>>();
        let pads_end = pads.iter().map(|(_, b)| *b).collect::<Vec<_>>();
        let mut cuda_pads_start = unsafe { self.device().alloc::<i64>(pads.len())? };
        self.device()
            .htod_copy_into(pads_start, &mut cuda_pads_start)?;
        let mut cuda_pads_end = unsafe { self.device().alloc::<i64>(pads.len())? };
        self.device().htod_copy_into(pads_end, &mut cuda_pads_end)?;

        let (kernel, reg_info) = load_ptx_and_get_data(
            "pad",
            &format!("pad_{}", T::STR),
            self.device(),
            self.device_cap(),
            &PAD,
        )
        .unwrap();

        let cfg = compute_kernel_launch_config(self.device(), &reg_info, res.size());

        let res_cuda_shape = res.cuda_shape()?;
        let res_cuda_strides = res.cuda_strides()?;
        let self_cuda_shape = self.cuda_shape()?;
        let self_cuda_strides = self.cuda_strides()?;
        unsafe {
            kernel.launch(
                cfg,
                (
                    res.cuda_slice(),
                    self.cuda_slice(),
                    val,
                    &res_cuda_shape,
                    &res_cuda_strides,
                    &self_cuda_shape,
                    &self_cuda_strides,
                    self.ndim(),
                    &cuda_pads_start,
                    &cuda_pads_end,
                    res.size(),
                ),
            )?
        };

        Ok(res)
    }

    fn topk(
        &self,
        _: i64,
        _: i64,
        _: bool,
        _: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError> {
        unimplemented!()
    }

    fn onehot(
        &self,
        _: usize,
        _: i64,
        _: Self::Meta,
        _: Self::Meta,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }

    fn scatter(
        &self,
        _: &Self::IndexOutput,
        _: i64,
        _: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}

impl<T, const DEVICE: usize, Al> HardMax<T> for _Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + Cmp<Output = bool>,
    bool: NormalOut<T> + Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cuda, DEVICE, Al>;
    fn hardmax(&self, _: i64) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}

impl<T, const DEVICE: usize, Al> TensorWhere for _Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cuda, DEVICE, Al>;
    type Condition = _Tensor<bool, Cuda, DEVICE, Al>;
    fn tensor_where(
        _: &Self::Condition,
        _: &Self::Output,
        _: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}
