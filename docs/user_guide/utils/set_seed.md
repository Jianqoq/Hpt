# set_seed

```rust
set_seed<B: BackendTy>(seed: u64)
```

Set the seed for random number generation
## Parameters:
`seed`: seed for generating random number
`B`: hpt::Cuda | hpt::Cpu


## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ❌         |
| Cuda    | ✅        |