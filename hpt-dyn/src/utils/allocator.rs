use hpt_allocator::{Cpu, traits::Allocator as AllocatorTrait};
#[cfg(feature = "cuda")]
use hpt_allocator::Cuda;
use hpt_common::{Pointer, error::base::TensorError};

use super::device::Device;

pub(crate) enum Allocator {
    BuiltinCpu(hpt_allocator::HptAllocator<Cpu>),
    #[cfg(feature = "cuda")]
    BuiltinCuda(hpt_allocator::HptAllocator<Cuda>),
}

fn alloc(
    allocator: &mut Allocator,
    layout: std::alloc::Layout,
    _device: &mut Device,
    cpu_alloc_method: fn(
        &hpt_allocator::HptAllocator<Cpu>,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<*mut u8, TensorError>,
    #[cfg(feature = "cuda")] cuda_alloc_method: fn(
        &hpt_allocator::HptAllocator<Cuda>,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<
        (*mut u8, std::sync::Arc<cudarc::driver::CudaDevice>),
        TensorError,
    >,
) -> Result<Pointer<u8>, TensorError> {
    match allocator {
        Allocator::BuiltinCpu(allocator) => {
            let res = (cpu_alloc_method)(allocator, layout, 0)?;
            Ok(Pointer::new(res, layout.size() as i64))
        }
        #[cfg(feature = "cuda")]
        Allocator::BuiltinCuda(allocator) => {
            let res = (cuda_alloc_method)(allocator, layout, _device.id())?;
            *_device = Device::CudaWithDevice(res.1);
            Ok(Pointer::new(res.0, layout.size() as i64))
        }
    }
}

impl Allocator {
    pub fn new(device: &Device) -> Result<Self, TensorError> {
        match device {
            Device::Cpu => Ok(Allocator::BuiltinCpu(hpt_allocator::HptAllocator::new())),
            #[cfg(feature = "cuda")]
            _ => Ok(Allocator::BuiltinCuda(hpt_allocator::HptAllocator::new())),
        }
    }

    pub(crate) fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        device: &mut Device,
    ) -> Result<Pointer<u8>, TensorError> {
        alloc(
            self,
            layout,
            device,
            hpt_allocator::HptAllocator::<Cpu>::allocate,
            #[cfg(feature = "cuda")]
            hpt_allocator::HptAllocator::<Cuda>::allocate,
        )
    }

    pub(crate) fn allocate_zeroed(
        &mut self,
        layout: std::alloc::Layout,
        device: &mut Device,
    ) -> Result<Pointer<u8>, TensorError> {
        alloc(
            self,
            layout,
            device,
            hpt_allocator::HptAllocator::<Cpu>::allocate_zeroed,
            #[cfg(feature = "cuda")]
            hpt_allocator::HptAllocator::<Cuda>::allocate_zeroed,
        )
    }
}
