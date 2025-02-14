# HPT

Hpt is a high performance N-dimensional array library. It is being highly optimized and is designed to be easy to use. Most of the operators are implemented based on Onnx operator list. Hence, you can use it to build most of the deep learning models.

# Features
#### Memory Layout
- Optimized memory layout with support for both contiguous and not contiguous tensors.
#### SIMD Support
- Leverages CPU SIMD instructions (SSE/AVX/NEON) for vectorized operations.
#### Iterator API
- Flexible iterator API for efficient element-wise/broadcast operations and custom implementations.
#### Multi-Threading
- Auto efficient parallel processing for CPU-intensive operations.
#### Broadcasting
- Automatic shape broadcasting for element-wise operations, similar to NumPy.
#### Type Safe
- Strong type system ensures correctness at compile time, preventing runtime errors.
#### Zero-Copy
- Minimizes memory overhead with zero-copy operations and efficient data sharing.
#### Auto Type Promote
- Allows auto type promote when compute with different types.
#### Custom Type
- Allows user to define their own data type for calculation (CPU support only).

# Note

Hpt is in early stage, bugs and wrong calculation results are expected

# Get Start
```rust
use hpt::*;

fn main() -> anyhow::Result<()> {
    let x = Tensor::new(&[1f64, 2., 3.]);
    let y = Tensor::new(&[4i64, 5, 6]);

    let result: Tensor<f64> = x + &y; // with `normal_promote` feature enabled, i64 + f64 will output f64
    println!("{}", result); // [5. 7. 9.]

    // All the available methods are listed in https://jianqoq.github.io/Hpt/user_guide/user_guide.html
    let result: Tensor<f64> = y.sin()?;
    println!("{}", result); // [-0.7568 -0.9589 -0.2794]
    Ok(())
}
```

To use Cuda, enable feature `cuda` (Note that Cuda is in development and not tested)
```rust
use hpt::*;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]).to_cuda::<0/*Cuda device id*/>()?;
    let y = Tensor::<i64>::new(&[4i64, 5, 6]).to_cuda::<0/*Cuda device id*/>()?;

    let result = x + &y; // with `normal_promote` feature enabled, i64 + f64 will output f64
    println!("{}", result); // [5. 7. 9.]

    // All the available methods are listed in https://jianqoq.github.io/Hpt/user_guide/user_guide.html
    let result: Tensor<f64, Cuda, 0> = y.sin()?;
    println!("{}", result); // [-0.7568 -0.9589 -0.2794]
    Ok(())
}
```

#### For more examples, reference [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-examples/examples) and [documentation](https://jianqoq.github.io/Hpt/user_guide/user_guide.html)

# How To Get Highest Performance
- Compile your program with the following configuration in `Cargo.toml`, note that `lto` is very important.
```cargo
opt-level = 3
lto = "fat"
codegen-units = 1
```
- Ensure your Env variable `RUSTFLAGS` enabled the best features your CPU has, like `-C target-feature=+avx2 -C target-feature=+fma`.

# Benchmarks
[benchmarks](https://jianqoq.github.io/Hpt/benchmarks/benchmarks.html)

# Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | 🚧        |

| CPU    | Supported |
|--------|-----------|
| AVX2   | ✅         |
| AVX512 | ❌         |
| SSE    | ✅         |
| Neon   | ✅         |

It is welcome to get contribution for supporting machines that is not supported in the list. Before contribute, please look at the [dev guide](https://jianqoq.github.io/Hpt/dev_guide/dev_guide.html).

# Documentations
For more details, visit https://jianqoq.github.io/Hpt/

# License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.