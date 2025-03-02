# set_display_elements

```rust
set_display_elements(lr_elements: usize)
```

Controls how many elements are shown for left handside and right handside for each dimension when displaying large tensors.
## Parameters:
- `lr_elements`: `usize`
  - Number of elements to display from left and right sides of each dimension
  - Must be a non-negative integer
  - Default is 3

## Examples
```rust
use hpt::{set_display_elements, Cpu, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::<i32, Cpu>::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Set to show 2 elements from each end
    set_display_elements(2);
    println!("{}", tensor); // Output: [1  2 ... 9 10 ]
    
    // Set to show 3 elements from each end
    set_display_elements(3);
    println!("{}", tensor); // Output: [1 2  3 ... 8 9 10 ]
    
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |