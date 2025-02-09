# Custom Type

Since hpt is designed in purely generic types, the user can define their own type and can use custom type to do all the computation hpt supports.

# How

You can reference the steps at [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-examples/examples/custom_type/main.rs).

# Note

Your custom type must implemented `Copy` trait. The reason why hpt doesn't support type with only `Clone` is because of the performance issue. For example, in convolution, we need to use fixed size array to preallocate registers `[T; N]`, and this requires `T` implemented `Copy` trait. If we allows the custom type implemented `Clone` trait only, we must preallocate registers using `vec![T; N]`, however, it is not sure if `Rust` will know we want to preallocate registers when it compiles but allocating with `[T; N]` may let `Rust` optimize and know we want to use registers easier.