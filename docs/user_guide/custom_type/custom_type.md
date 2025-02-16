# Custom Type

Since hpt is designed in purely generic types, the user can define their own type and can use custom type to do all the computation hpt supports.

# How

You can reference the steps at [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-examples/examples/custom_type/main.rs).

# Note

For now, your custom type must implemented `Copy` trait. The reason why hpt doesn't support type with only `Clone` is because of the conv2d implementation issue. The conv2d used fixed size array to preallocate registers `[T; N]`, and this requires `T` implemented `Copy` trait.

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |