use tensor_common::{ layout::Layout, pointer::Pointer };

/// This struct is the heart of the `DiffTensors` and `BasicTensors`. Both of them are just `wrappers` around this struct.
///
/// All the operations are happen on this struct.
///
/// The `DiffTensors` and `BasicTensors` are just call `_Tensor` methods and add extra logic.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor.
///
///  If the tensor is a view of another tensor, the parent tensor will be the original tensor.
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory.
pub struct _Tensor<T> {
    data: Pointer<T>,
    parent: Option<Pointer<T>>,
    layout: Layout,
    mem_layout: std::alloc::Layout,
}

impl<T> _Tensor<T> {}
