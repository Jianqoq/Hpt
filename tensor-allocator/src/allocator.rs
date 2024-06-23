use std::{ alloc::Layout, num::NonZeroUsize, sync::Mutex };

use hashbrown::HashSet;
use lru::LruCache;
use once_cell::sync::Lazy;

static mut CACHE: Lazy<Allocator> = Lazy::new(|| Allocator::new(1000));

struct Allocator {
    allocator: Mutex<_Allocator>,
}

impl Allocator {
    pub fn allocate(layout: Layout) -> *mut u8 {
        unsafe { CACHE.allocator.lock().unwrap().allocate(layout) }
    }

    pub fn deallocate(ptr: *mut u8, layout: Layout) {
        unsafe { CACHE.allocator.lock().unwrap().deallocate(ptr, layout) }
    }
}

impl Allocator {
    pub fn new(capacity: usize) -> Self {
        Allocator {
            allocator: Mutex::new(_Allocator {
                cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
                allocated: HashSet::new(),
                capacity,
            }),
        }
    }
}

struct _Allocator {
    cache: LruCache<Layout, Vec<*mut u8>>,
    allocated: HashSet<*mut u8>,
    capacity: usize,
}

impl _Allocator {
    fn allocate(&mut self, layout: Layout) -> *mut u8 {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout) {
            if let Some(ptr) = ptr.pop() {
                ptr
            } else {
                let ptr = unsafe { std::alloc::alloc(layout) };
                self.allocated.insert(ptr);
                ptr
            }
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
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
        ptr
    }

    fn deallocate(&mut self, ptr: *mut u8, layout: Layout) {
        self.allocated.remove(&ptr);
        if let Some(ptrs) = self.cache.get_mut(&layout) {
            ptrs.push(ptr);
        } else {
            self.cache.put(layout, vec![ptr]);
        }
    }
}
