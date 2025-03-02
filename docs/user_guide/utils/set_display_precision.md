# set_display_precision

```rust
set_display_precision(precision: usize)
```

Controls how many decimal places are shown when displaying tensor values.
## Parameters:
- `precision`: `usize`
  - Number of decimal places to display
  - Must be a non-negative integer
  - Default is 4

## Examples
```rust
use hpt::{set_display_precision, Cpu, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Set precision to 3 decimal places
    set_display_precision(3);

    let tensor = Tensor::<f64, Cpu>::new([1.23456789]);
    println!("{}", tensor); // Output: 1.235

    // Set precision to 6 decimal places
    set_display_precision(6);
    println!("{}", tensor); // Output: 1.234568
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |