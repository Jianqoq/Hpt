#![allow(unused)]

use hpt::{backend::Cpu, error::TensorError, ops::TensorCreator, Tensor};

#[derive(Clone)]
struct CustomCpuAllocator;
#[derive(Clone)]
struct CustomCudaAllocator;

impl hpt::alloc::Allocator for CustomCpuAllocator {
    type Output = *mut u8;

    type CpuAllocator = CustomCpuAllocator;

    #[cfg(feature = "cuda")]
    type CudaAllocator = CustomCudaAllocator;

    fn allocate(
        &self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert_eq!(ptr as usize % layout.align(), 0); // you must make sure the memory is aligned
        println!("allocate memory");
        Ok(ptr)
    }

    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError> {
        println!("allocate zeroed memory");
        self.allocate(layout, device_id)
    }

    fn deallocate(
        &self,
        ptr: *mut u8,
        layout: &std::alloc::Layout,
        should_drop: bool,
        device_id: usize,
    ) {
        assert_eq!(ptr as usize % layout.align(), 0); // you must make sure the memory is aligned
        unsafe {
            std::alloc::dealloc(ptr, *layout);
        }
        println!("deallocate memory");
    }

    // store the ptr at somewhere, like global variable
    fn insert_ptr(&self, ptr: *mut u8, device_id: usize) {
        println!("insert ptr to cpu allocator");
    }

    // clear all the memory allocated, this method will be called when the program exits
    fn clear(&self) {
        println!("clear cpu allocator");
    }

    // create a new allocator
    fn new() -> Self {
        Self {}
    }

    /// forget the ptr from the allocator
    fn forget(&self, ptr: *mut u8, device_id: usize) {
        println!("forget ptr from cpu allocator");
    }
}

impl hpt::alloc::Allocator for CustomCudaAllocator {
    type Output = *mut u8;

    type CpuAllocator = CustomCpuAllocator;

    #[cfg(feature = "cuda")]
    type CudaAllocator = CustomCudaAllocator;

    fn allocate(
        &self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError> {
        // allocate memory on cuda
        todo!()
    }

    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError> {
        todo!()
    }

    fn deallocate(
        &self,
        ptr: *mut u8,
        layout: &std::alloc::Layout,
        should_drop: bool,
        device_id: usize,
    ) {
        // deallocate memory on cuda
        todo!()
    }

    // store the ptr at somewhere, like global variable
    fn insert_ptr(&self, ptr: *mut u8, device_id: usize) {
        todo!()
    }

    // clear all the memory allocated, this method will be called when the program exits
    fn clear(&self) {
        todo!()
    }

    // create a new allocator
    fn new() -> Self {
        Self {}
    }

    /// forget the ptr from the allocator
    fn forget(&self, ptr: *mut u8, device_id: usize) {
        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    // // new method won't call allocate method, it will call insert_ptr method
    let a = Tensor::<i32, Cpu, 0, CustomCpuAllocator>::new(&[1, 2, 3, 4]);
    // allocate method will be called
    let b = Tensor::<i32, Cpu, 0, CustomCpuAllocator>::arange(0, 100)?;
    println!("{:?}", a);
    println!("{:?}", b);
    Ok(())
}
