#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "cuda")]
    CudaWithDevice(std::sync::Arc<cudarc::driver::CudaDevice>),
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => self.id() == other.id(),
            #[cfg(feature = "cuda")]
            (Device::Cpu, Device::Cuda(_)) => false,
            #[cfg(feature = "cuda")]
            (Device::Cpu, Device::CudaWithDevice(_)) => false,
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), Device::Cpu) => false,
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), Device::Cuda(_)) => self.id() == other.id(),
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), Device::CudaWithDevice(_)) => self.id() == other.id(),
            #[cfg(feature = "cuda")]
            (Device::CudaWithDevice(_), Device::Cpu) => false,
            #[cfg(feature = "cuda")]
            (Device::CudaWithDevice(_), Device::Cuda(_)) => self.id() == other.id(),
            #[cfg(feature = "cuda")]
            (Device::CudaWithDevice(_), Device::CudaWithDevice(_)) => self.id() == other.id(),
        }
    }
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
