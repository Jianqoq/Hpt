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
