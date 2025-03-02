# resize_cpu_lru_cache

```rust
resize_cpu_lru_cache(new_size: usize, device_id: usize)
```

Adjusts the size of the memory cache used by the CPU allocator. This function allows dynamic control over memory caching behavior.

## Parameters:
- `new_size`: `usize`
  - New capacity of the LRU cache
  - Must be a non-negative integer
  - Determines how many memory allocations can be cached
- `device_id`: `usize`
  - ID of the CPU device
  - Usually 0 for single CPU systems

## Behavior
- When `new_size` >= current size:
  - Cache capacity increases
  - Existing cached memory is preserved
- When `new_size` < current size:
  - All cached memory is deallocated
  - Cache is resized to new capacity

## Examples
```rust
use hpt::{resize_cpu_lru_cache, TensorError};

fn main() -> Result<(), TensorError> {
    // Increase cache size to 1000 entries
    resize_cpu_lru_cache(1000, 0);
    
    // Reduce cache size to free memory
    resize_cpu_lru_cache(100, 0);
    
    Ok(())
}
```