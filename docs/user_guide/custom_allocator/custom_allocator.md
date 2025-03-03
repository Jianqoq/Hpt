# Custom Allocator

Since hpt is designed in purely generic types, the user can define their own memory allocator.

# How

You can reference the steps at [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-examples/examples/custom_allocator/main.rs).

# Note

Custom Allocator must have life time `'static`, `Send`, `Sync`, `Clone` implemented

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |