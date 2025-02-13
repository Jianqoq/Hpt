use cudarc::driver::DeviceRepr;

#[repr(C)]
pub(crate) struct CudaSlice {
    pub(crate) inner: cudarc::driver::sys::CUdeviceptr,
}

unsafe impl DeviceRepr for CudaSlice {}
