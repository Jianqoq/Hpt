use std::num::NonZeroUsize;

use lru::LruCache;

use crate::ptr::SafePtr;

pub fn resize_lru_cache(
    cache: &mut LruCache<std::alloc::Layout, Vec<SafePtr>>,
    deallocate_fn: impl Fn(*mut u8, std::alloc::Layout),
    new_size: usize,
) {
    if cache.cap().get() <= new_size {
        cache.resize(NonZeroUsize::new(new_size).unwrap());
    } else {
        let new = LruCache::new(NonZeroUsize::new(new_size).unwrap());
        for (layout, ptrs) in cache.iter() {
            for safe_ptr in ptrs {
                deallocate_fn(safe_ptr.ptr, *layout);
            }
        }
        *cache = new;
    }
}
