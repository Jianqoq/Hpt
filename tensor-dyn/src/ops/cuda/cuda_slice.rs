use cudarc::driver::DeviceRepr;

#[repr(transparent)]
pub(crate) struct CudaSlice {
    pub(crate) inner: cudarc::driver::sys::CUdeviceptr,
}

unsafe impl DeviceRepr for CudaSlice {}
unsafe impl DeviceRepr for &mut CudaSlice {}
unsafe impl DeviceRepr for &CudaSlice {}
