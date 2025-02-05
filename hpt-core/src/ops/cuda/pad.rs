use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::{tensor_base::_Tensor, Cuda, Tensor};
use cudarc::driver::{DeviceRepr, LaunchAsync};
use hpt_cudakernels::PAD;
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};

use super::cuda_utils::compute_kernel_launch_config;
impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn pad(&self, pads: &[(i64, i64)], val: T) -> anyhow::Result<_Tensor<T, Cuda, DEVICE_ID>> {
        let res_shape = self
            .shape()
            .iter()
            .zip(pads.iter())
            .map(|(x, (a, b))| x + a + b)
            .collect::<Vec<_>>();

        let res = _Tensor::<T, Cuda, DEVICE_ID>::full(val, &res_shape)?;
        let mut pads = pads.to_vec();
        if pads.len() < self.ndim() {
            pads.resize(self.ndim(), (0, 0));
        }
        let pads_start = pads.iter().map(|(a, _)| *a).collect::<Vec<_>>();
        let pads_end = pads.iter().map(|(_, b)| *b).collect::<Vec<_>>();
        println!("{:?}", pads_start);
        println!("{:?}", pads_end);
        let mut cuda_pads_start = unsafe { self.device().alloc::<i64>(pads.len())? };
        self.device()
            .htod_copy_into(pads_start, &mut cuda_pads_start)?;
        let mut cuda_pads_end = unsafe { self.device().alloc::<i64>(pads.len())? };
        self.device().htod_copy_into(pads_end, &mut cuda_pads_end)?;

        let (kernel, reg_info) = load_ptx_and_get_data(
            "pad",
            &format!("pad_{}", T::ID),
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
}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID> {
    /// Converts the input tensor into a one-hot encoded tensor along a specified axis.
    ///
    /// This method transforms the input tensor into a one-hot encoded format, where the values
    /// along the specified axis are converted into vectors of size `depth`. Each vector contains
    /// a `true_val` at the index specified by the input tensor and `false_val` elsewhere.
    ///
    /// # Arguments
    ///
    /// * `depth` - The size of the one-hot vectors. This represents the number of unique categories
    ///   for the one-hot encoding. Each element in the input tensor will be transformed into a one-hot
    ///   vector of this length.
    /// * `axis` - The axis along which the one-hot encoding is applied. If the axis is negative, it is
    ///   treated as counting from the last dimension of the tensor. The new one-hot vectors will be inserted
    ///   along this axis.
    /// * `_true_val` - The value that will be placed at the position corresponding to the one-hot index (usually 1).
    /// * `false_val` - The value that will fill the other positions in the one-hot vector (usually 0).
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the one-hot encoded values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn pad(&self, pads: &[(i64, i64)], val: T) -> anyhow::Result<Tensor<T, Cuda, DEVICE_ID>> {
        Ok(self.inner.pad(pads, val)?.into())
    }
}
