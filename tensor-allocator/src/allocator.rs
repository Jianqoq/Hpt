use std::{ alloc::Layout, hash::Hash, num::NonZeroUsize, ops::Deref, sync::{ Arc, Mutex } };

use hashbrown::{ HashMap, HashSet };
use lru::LruCache;
use once_cell::sync::Lazy;
use wgpu::{ Buffer, BufferUsages, Device };

use crate::{ strorage::CPU_STORAGE, WGPU_STORAGE };

pub static mut CACHE: Lazy<Allocator> = Lazy::new(|| Allocator::new(1000));
pub static mut WGPU_CACHE: Lazy<WgpuAllocator> = Lazy::new(|| WgpuAllocator::new(1000));

pub struct Allocator {
    allocator: Mutex<_Allocator>,
}

impl Allocator {
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        self.allocator.lock().unwrap().allocate(layout)
    }

    pub fn deallocate(&self, ptr: *mut u8, layout: &Layout) {
        self.allocator.lock().unwrap().deallocate(ptr, layout);
    }

    pub fn insert_ptr(&self, ptr: *mut u8) {
        self.allocator.lock().unwrap().insert_ptr(ptr);
    }

    pub fn clear(&self) {
        let mut allocator = self.allocator.lock().unwrap();
        for (layout, ptrs) in allocator.cache.iter_mut() {
            for ptr in ptrs.iter() {
                unsafe {
                    std::alloc::dealloc(*ptr, layout.clone());
                }
            }
        }
        allocator.cache.clear();
        assert_eq!(allocator.allocated.len(), 0);
    }
}

impl Allocator {
    pub fn new(capacity: usize) -> Self {
        Allocator {
            allocator: Mutex::new(_Allocator {
                cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
                allocated: HashSet::new(),
            }),
        }
    }
}

struct _Allocator {
    cache: LruCache<Layout, Vec<*mut u8>>,
    allocated: HashSet<*mut u8>,
}

impl _Allocator {
    fn allocate(&mut self, layout: Layout) -> *mut u8 {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout) {
            if let Some(ptr) = ptr.pop() {
                ptr
            } else {
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    panic!("Failed to allocate memory, for {} MB", layout.size() / 1024 / 1024);
                }
                self.allocated.insert(ptr);
                ptr
            }
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate memory, for {} MB", layout.size() / 1024 / 1024);
            }
            self.allocated.insert(ptr);
            ptr
        };
        if self.cache.cap().get() == self.cache.len() {
            if let Some((layout, ptrs)) = self.cache.pop_lru() {
                for ptr in ptrs {
                    unsafe {
                        std::alloc::dealloc(ptr, layout);
                    }
                }
            }
        }
        // println!("Allocating ptr {:p}", ptr);
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    *cnt += 1;
                } else {
                    storage.insert(ptr, 1);
                }
            }
        }
        ptr
    }

    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout) {
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    // println!("Deallocating ptr {:p}", ptr);
                    *cnt -= 1;
                    if *cnt == 0 {
                        self.allocated.remove(&ptr);
                        if let Some(ptrs) = self.cache.get_mut(layout) {
                            ptrs.push(ptr);
                        } else {
                            self.cache.put(layout.clone(), vec![ptr]);
                        }
                    }
                } else {
                    panic!("ptr {:p} not found in storage", ptr);
                }
            }
        }
    }

    fn insert_ptr(&mut self, ptr: *mut u8) {
        self.allocated.insert(ptr);
        // println!("Inserting ptr {:p}", ptr);
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    *cnt += 1;
                } else {
                    storage.insert(ptr, 1);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct BufferWrapper {
    pub buffer: Arc<Buffer>,
}

impl PartialEq for BufferWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.buffer.global_id() == other.buffer.global_id()
    }
}

impl Eq for BufferWrapper {}

impl Hash for BufferWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.buffer.global_id().hash(state);
    }
}

#[derive(Clone)]
pub struct DeviceWrapper {
    pub device: Arc<Device>,
}

impl Deref for DeviceWrapper {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl PartialEq for DeviceWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.device.global_id() == other.device.global_id()
    }
}

impl Eq for DeviceWrapper {}

impl Hash for DeviceWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.device.global_id().hash(state);
    }
}

pub struct WgpuAllocator {
    allocator: Mutex<_WgpuAllocator>,
}

impl WgpuAllocator {
    pub fn allocate(
        &self,
        layout: &Layout,
        device: &DeviceWrapper,
        mapped_at_creation: bool
    ) -> BufferWrapper {
        self.allocator.lock().unwrap().allocate(layout, device, mapped_at_creation)
    }

    pub fn deallocate(&self, device: &DeviceWrapper, buffer: &BufferWrapper, layout: &Layout) {
        self.allocator.lock().unwrap().deallocate(device, buffer, layout);
    }
}

impl WgpuAllocator {
    pub fn new(capacity: usize) -> Self {
        WgpuAllocator {
            allocator: Mutex::new(_WgpuAllocator {
                devices: HashMap::new(),
                capacity,
            }),
        }
    }
}

struct _WgpuAllocator {
    devices: HashMap<DeviceWrapper, _WgpuAllocatorHelper>,
    capacity: usize,
}

impl _WgpuAllocator {
    fn allocate(
        &mut self,
        layout: &Layout,
        device: &DeviceWrapper,
        mapped_at_creation: bool
    ) -> BufferWrapper {
        if let Some(allocator) = self.devices.get_mut(device) {
            let buffer = allocator.allocate(
                layout,
                device,
                wgpu::BufferUsages::COPY_DST |
                    wgpu::BufferUsages::COPY_SRC |
                    wgpu::BufferUsages::STORAGE,
                mapped_at_creation
            );
            // println!("allocated buffer, {:?}", buffer.buffer.global_id());
            buffer
        } else {
            let mut allocator = _WgpuAllocatorHelper {
                cache: LruCache::new(NonZeroUsize::new(self.capacity).unwrap()),
                allocated: HashSet::new(),
            };
            let buffer = allocator.allocate(
                layout,
                device,
                wgpu::BufferUsages::COPY_DST |
                    wgpu::BufferUsages::COPY_SRC |
                    wgpu::BufferUsages::STORAGE,
                mapped_at_creation
            );
            // println!("allocated buffer, {:?}", buffer.buffer.global_id());
            self.devices.insert(device.clone(), allocator);
            buffer
        }
    }

    fn deallocate(&mut self, device: &DeviceWrapper, buffer: &BufferWrapper, layout: &Layout) {
        if let Some(ptr) = self.devices.get_mut(device) {
            ptr.deallocate(buffer, layout);
        }
    }
}

struct _WgpuAllocatorHelper {
    cache: LruCache<Layout, Vec<BufferWrapper>>,
    allocated: HashSet<BufferWrapper>,
}

impl _WgpuAllocatorHelper {
    fn allocate(
        &mut self,
        layout: &Layout,
        device: &DeviceWrapper,
        usage: BufferUsages,
        mapped_at_creation: bool
    ) -> BufferWrapper {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout) {
            if let Some(buffer) = ptr.pop() {
                let usage = buffer.buffer.usage();
                if usage == buffer.buffer.usage() {
                    buffer
                } else {
                    cache(self, device, layout, usage, mapped_at_creation)
                }
            } else {
                cache(self, device, layout, usage, mapped_at_creation)
            }
        } else {
            cache(self, device, layout, usage, mapped_at_creation)
        };

        fn cache(
            helper: &mut _WgpuAllocatorHelper,
            device: &DeviceWrapper,
            layout: &Layout,
            usage: BufferUsages,
            mapped_at_creation: bool
        ) -> BufferWrapper {
            let buffer = device.device.create_buffer(
                &(wgpu::BufferDescriptor {
                    label: Some("Tensor buffer"),
                    size: layout.size() as u64,
                    usage,
                    mapped_at_creation,
                })
            );
            let buffer = Arc::new(buffer);
            helper.allocated.insert(BufferWrapper { buffer: buffer.clone() });
            BufferWrapper { buffer }
        }
        unsafe {
            if let Ok(mut storage) = WGPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr.buffer.global_id()) {
                    *cnt += 1;
                } else {
                    storage.insert(ptr.buffer.global_id(), 1);
                }
            }
        }
        if self.cache.cap().get() == self.cache.len() {
            if let Some((_, ptrs)) = self.cache.pop_lru() {
                for ptr in ptrs {
                    if Arc::strong_count(&ptr.buffer) == 1 {
                        ptr.buffer.unmap();
                    } else {
                        panic!("Buffer still in use");
                    }
                }
            }
        }
        ptr
    }

    fn deallocate(&mut self, buffer: &BufferWrapper, layout: &Layout) {
        unsafe {
            if let Ok(mut storage) = WGPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&buffer.buffer.global_id()) {
                    *cnt -= 1;
                    if *cnt == 0 {
                        // println!("Deallocating buffer, {:?}", buffer.buffer.global_id());
                        self.allocated.remove(buffer);
                        if let Some(ptrs) = self.cache.get_mut(layout) {
                            ptrs.push(buffer.clone());
                        } else {
                            self.cache.put(layout.clone(), vec![buffer.clone()]);
                        }
                    }
                } else {
                    panic!("Buffer not found in storage");
                }
            }
        }
    }
}
