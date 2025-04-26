#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "cuda")]
    CudaWithDevice(std::sync::Arc<cudarc::driver::CudaDevice>),
}

impl Device {
    pub fn id(&self) -> usize {
        match self {
            Device::Cpu => 0,
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => *id,
            #[cfg(feature = "cuda")]
            Device::CudaWithDevice(device) => device.ordinal(),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => write!(f, "cuda({})", id),
            #[cfg(feature = "cuda")]
            Device::CudaWithDevice(device) => write!(f, "cuda({})", device.ordinal()),
        }
    }
}
