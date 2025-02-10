# Save

To save a list of Tensors

1. Create saver
```rust
let saver = TensorSaver::new("path/to/save/file");
```

2. Push Tensors to the saver. Note: Tensors can be different type.
```rust
let a = Tensor::<f32>::randn([1, 100, 100, 100])?;
// push method will move the saver and will return saver back.
let saver = saver.push(
    /*name of the tensor to save*/ "a",
    /*tensor to save*/ a,
    hpt_core::CompressionAlgo::Gzip,
    hpt_core::Endian::Little,
    /*compression level*/ 9,
);
```
3. Save
```rust
saver.save()?;
```

Final code will be

```rust
use hpt_core::{Random, Tensor, TensorSaver};
fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::randn([1, 100, 100, 100])?;
    let saver = TensorSaver::new("path/to/save/file");
    let saver = saver.push(
        "a",
        a,
        hpt_core::CompressionAlgo::Gzip,
        hpt_core::Endian::Little,
        9,
    );
    saver.save()?;
    Ok(())
}
```

# Load

To load a list of Tensors, you must make sure the Tensors you load are in same type.

1. Create a loader
```rust
let loader = TensorLoader::new("path/to/load/file");
```

2. Load Tensors by passing the tensor name and slice.
```rust
let loaded: std::collections::HashMap<String, Tensor<f32>> = loader
    .push("a", &match_selection![:, 1:-1:2, 3::, :]) // you can use python slice syntax
    .push("b", &[]) // if you don't slice, simply use empty slice
    .load::<f32, Tensor<f32>, { std::mem::size_of::<f32>() }>()?;
```

3. If your file contains different type of Tensors, load it multiple times.

# Save Struct

If you have a struct that contains a Tensor and you want to save a whole struct.

1. Use `Save` and `Load` Derive macro.
```rust
use hpt_core::{Load, Save};
```

2. Derive the `Save` and `Load` for your struct.
```rust
#[derive(Save, Load)]
pub struct Linear {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}
```

3. If you want to save a Tensor with specific compression
```rust
#[derive(Save, Load)]
pub struct Linear {
    #[compress(algo = "gzip", level = "9", endian = "little")]
    weight: Tensor<f32>,
    #[compress(algo = "zlib", level = "5", endian = "big")]
    bias: Tensor<f32>,
}
```

4. Save the struct
```rust
impl Linear {
    pub fn new(in_features: i64, out_features: i64) -> Result<Self, TensorError> {
        Ok(Self {
            weight: Tensor::<f32>::randn([in_features, out_features])?,
            bias: Tensor::<f32>::zeros([out_features])?,
        })
    }
}
let linear = Linear::new(1000, 1000)?;
linear.save("path/to/save/linear")?;
```

# Load Struct

If you saved a struct in a file and you want to load the struct

1. Make sure your struct Derived `Load`.

2. Load the file
```rust
let linear = Linear::load("path/to/save/linear")?;
```