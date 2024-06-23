use tensor_common::{ layout::Layout, pointer::Pointer, shape::Shape, strides::Strides };
use tensor_types::dtype::TypeCommon;

pub trait TensorInfo<T> {
    fn ptr(&self) -> Pointer<T>;
    fn size(&self) -> usize;
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &Strides;
    fn layout(&self) -> &Layout;
    fn parent(&self) -> Option<Pointer<T>>;
    fn ndim(&self) -> usize;
    fn is_contiguous(&self) -> bool;
    fn elsize() -> usize {
        std::mem::size_of::<T>()
    }
}

pub trait TensorAlloc<Output = Self> {
    type Meta;
    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Output> where Self: Sized;
}

pub trait CommonBounds: Sync + Send + Clone + Copy + TypeCommon + 'static {}
impl<T: Sync + Send + Clone + Copy + TypeCommon + 'static> CommonBounds for T {}
