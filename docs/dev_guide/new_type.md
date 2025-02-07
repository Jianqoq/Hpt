# How to add new type support

### Things to know
1. Hpt has vector and scalar implementation. To support new type, you will need to go to `hpt-types` crate.

### How
1. Go to `hpt-types/src/dtype.rs`.
2. Implement `TypeCommon` for new type, implement `CudaType` for new type if the new type is supported in cuda
3. Add Dtype variant for the `Dtype` enum.
4. Go to `hpt-types/src/scalars/`, based on the existing implementation for primitive type, implement the traits for new type.
5. Go to `hpt-types/src/vectors/`, based on the existing implementation for primitive type, implement the traits for new type.
6. Go to `hpt-types/src/convertion.rs`, add type conversion for the new type, add `to_new_type` method in traits.
7. Go to `hpt-types/src/promotion/`, add type promotion for new type.