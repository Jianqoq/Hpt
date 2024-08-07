use half::f16;
use crate::tensor_base::_Tensor;
use tensor_common::shape::Shape;
use tensor_common::pointer::Pointer;
use std::alloc::Layout;
use std::mem::ManuallyDrop;
use tensor_traits::tensor::TensorCreator;
use std::sync::Arc;
use tensor_common::strides_utils::shape_to_strides;
use tensor_allocator::CACHE;
use num::complex::{ Complex32, Complex64 };
use half::bf16;
use crate::backend::TensorBackend;
use crate::backend::Backend;

macro_rules! from_scalar {
    ($($t:ident),*) => {
        $(impl Into<_Tensor<$t>> for $t {
            fn into(self) -> _Tensor<$t> {
                let ret = _Tensor::<$t>::empty(vec![]).unwrap();
                ret.as_raw_mut()[0] = self;
                return ret;
            }
        })*
    };
}

macro_rules! impl_type_num {
    (num, $($t:ident),*) => {
        $(
            impl TypeNum for $t {
                fn type_num() -> Dtype {
                    return map_type_num!($t);
                }
            }
        )*
    };

    (vec, $($t:ident),*) => {
        $(
            impl From<Vec<$t>> for _Tensor<$t> {
                fn from(data: Vec<$t>) -> Self {
                    let mut ptr = data.as_ptr() as *mut $t;
                    let length = data.len();
                    let res_shape = Shape::from(vec![length as i64]);
                    let layout;
                    if (ptr as usize) % 32 == 0 {
                        let _ = ManuallyDrop::new(data);
                        layout = Layout::from_size_align(length * std::mem::size_of::<$t>(), 32).unwrap();
                    } else {
                        layout = Layout::from_size_align(data.len() * std::mem::size_of::<$t>(), 32).unwrap();
                        ptr = unsafe { CACHE.allocate(layout) } as *mut $t;
                        unsafe {
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                        }
                    }
                    return _Tensor {
                        data: Pointer::new(ptr),
                        parent: None,
                        layout: tensor_common::layout::Layout::new(res_shape, vec![1]),
                        mem_layout: Arc::new(layout),
                        _backend: Backend::new(ptr as u64),
                    };
                }
            }
        )*
    };
    (ndarray, $($generic:ident),*; $($vars:ident),*; $ct:ident, $($t:ident),*) => {
            impl<$(const $generic: usize), *> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct> {
                fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                    let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                    let shape = Shape::from(vec![$($generic as i64), *]);

                    repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element));
                    let mut ptr = vec.as_mut_ptr();
                    let length = repeate_generic!(mul, $($generic), *);
                    let layout;
                    if (ptr as usize) % 32 == 0 {
                        let _ = ManuallyDrop::new(vec);
                        layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                    } else {
                        layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                        ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                        unsafe {
                            for i in 0..vec.len() {
                                std::ptr::write(ptr.add(i), vec[i]);
                            }
                        }
                    }
                    let strides = shape_to_strides(&shape);
    
                    return _Tensor {
                        data: Pointer::new(ptr),
                        parent: None,
                        layout: tensor_common::layout::Layout::new(shape, strides),
                        mem_layout: Arc::new(layout),
                        _backend: Backend::new(ptr as u64),
                    };
                }
            }
            impl_type_num!(ndarray, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
        impl<$(const $generic: usize), *> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct> {
            fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                if (ptr as usize) % 32 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                    ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                    unsafe {
                        for i in 0..vec.len() {
                            std::ptr::write(ptr.add(i), vec[i]);
                        }
                    }
                }
                let strides = shape_to_strides(&shape);

                return _Tensor {
                    data: Pointer::new(ptr),
                    parent: None,
                    layout: tensor_common::layout::Layout::new(shape, strides),
                    mem_layout: Arc::new(layout),
                    _backend: Backend::new(ptr as u64),
                };
            }
        }
    };

    (
        ndarray_source_target,
        $source:ident,
        $($generic:ident),*;
        $($vars:ident),*;
        $ct:ident,
        $($t:ident),*
    ) => {
        impl<$(const $generic: usize), *> From<repeate_generic!(nested_array_type, $($generic), *; $source)> for _Tensor<$ct> {
            fn from(data: repeate_generic!(nested_array_type, $($generic), *; $source)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = vec![$($generic as i64), *];

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element.into()));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                if (ptr as usize) % 32 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                    ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                    unsafe {
                        for i in 0..vec.len() {
                            std::ptr::write(ptr.add(i), vec[i]);
                        }
                    }
                }
                let strides = shape_to_strides(&shape);

                return _Tensor {
                    data: Pointer::new(ptr),
                    parent: None,
                    layout: tensor_common::layout::Layout::new(shape.into(), strides),
                    mem_layout: Arc::new(layout),
                };
            }
        }
        impl_type_num!(ndarray_source_target, $source, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray_source_target, $source:ident, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
    impl<$(const $generic: usize), *> From<repeate_generic!(nested_array_type, $($generic), *; $source)> for _Tensor<$ct> {
        fn from(data: repeate_generic!(nested_array_type, $($generic), *; $source)) -> Self {
            let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
            let shape = Shape::from(vec![$($generic as i64), *]);

            repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element.into()));
            let mut ptr = vec.as_mut_ptr();
            let length = repeate_generic!(mul, $($generic), *);
            let layout;
            if (ptr as usize) % 32 == 0 {
                let _ = ManuallyDrop::new(vec);
                layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
            } else {
                layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                unsafe {
                    for i in 0..vec.len() {
                        std::ptr::write(ptr.add(i), vec[i]);
                    }
                }
            }
            let strides = shape_to_strides(&shape);
            return _Tensor {
                data: Pointer::new(ptr),
                parent: None,
                layout: tensor_common::layout::Layout::new(shape, strides),
                mem_layout: Arc::new(layout),
                _backend: Backend::new(ptr as u64),
            };
        }
    }
    };

    (ndarray_ref, $($generic:ident),*; $($vars:ident),*; $ct:ident, $($t:ident),*) => {
        impl<$(const $generic: usize), *> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct> {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(*element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                if (ptr as usize) % 32 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                    ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                    unsafe {
                        for i in 0..vec.len() {
                            std::ptr::write(ptr.add(i), vec[i]);
                        }
                    }
                }
                let strides = shape_to_strides(&shape);

                return _Tensor {
                    data: Pointer::new(ptr),
                    parent: None,
                    layout: tensor_common::layout::Layout::new(shape, strides),
                    mem_layout: Arc::new(layout),
                    _backend: Backend::new(ptr as u64),
                };
            }
        }
        impl_type_num!(ndarray_ref, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray_ref, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
        impl<$(const $generic: usize), *> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct> {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(*element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                if (ptr as usize) % 32 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 32).unwrap();
                    ptr = unsafe { CACHE.allocate(layout) } as *mut $ct;
                    unsafe {
                        for i in 0..vec.len() {
                            std::ptr::write(ptr.add(i), vec[i]);
                        }
                    }
                }
                let strides = shape_to_strides(&shape);

                return _Tensor {
                    data: Pointer::new(ptr),
                    parent: None,
                    layout: tensor_common::layout::Layout::new(shape, strides),
                    mem_layout: Arc::new(layout),
                    _backend: Backend::new(ptr as u64),
                };
            }
        }
    };
}

#[macro_export]
macro_rules! repeate_generic {
    (const, $($t:ident),*) => {
        impl<$(const $t: usize), *>
    };
    (nested_array, $n:expr, $($t:expr),*; $data_type:ident) => {
        [repeate_generic!(nested_array, $($t), *; $data_type);$n];
    };
    (nested_array, $t:expr; $data_type:ident) => {
        [$data_type; $t]
    };
    (nested_array_type, $n:expr, $($t:expr),*; $data_type:ident) => {
        [repeate_generic!(nested_array_type, $($t), *; $data_type);$n]
    };
    (nested_array_type, $t:expr; $data_type:ident) => {
        [$data_type; $t]
    };
    (operations, $op:tt, $n:expr, $($t:expr),*) => {
        $n $op repeate_generic!(operations, $op, $($t), *)
    };
    (operations, $op:tt, $n:expr) => {
        $n
    };
    (iterate, $data:ident; $vec:ident; $n:ident, $($t:ident),*) => {
        $data.into_iter().flat_map(|$n| repeate_generic!(iterate, $vec; $n;; $($t), *))
    };
    (iterate, $data:ident; $vec:ident; $n:ident) => {
        $data.into_iter().flat_map(|$n| repeate_generic!(iterate, $vec; $n;;))
    };
    (iterate, $vec:ident; $n:ident; ; $n2:ident, $($t:ident),*) => {
        $n.into_iter().flat_map(|$n2| repeate_generic!(iterate, $vec; $n2;; $($t), *))
    };
    (iterate, $vec:ident; $n:ident; ; $n2:ident) => {
        $n.into_iter().flat_map(|$n2| repeate_generic!(iterate, $vec; $n2;;))
    };
    (iterate, $vec:ident; $n:ident; ;) => {
        $n.into_iter()
    };
    (iterate, $data:ident; $vec:ident;) => {
        $data.into_iter()
    };
    (mul, $n:expr, $($t:expr),*) => {
        $n * repeate_generic!(mul, $($t), *)
    };
    (mul, $n:expr) => {
        $n
    };
}

from_scalar!(bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, bf16, f32, f64, Complex32, Complex64);
impl_type_num!(vec, bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64); // prettier-ignore
impl_type_num!(ndarray, N; ; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N; ; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M; i; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M; i; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O; i, j; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O; i, j; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O, P; i, j, k; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O, P; i, j, k; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O, P, Q; i, j, k, l; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O, P, Q; i, j, k, l; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O, P, Q, R; i, j, k, l, m; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O, P, Q, R; i, j, k, l, m; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O, P, Q, R, S; i, j, k, l, m, n; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O, P, Q, R, S; i, j, k, l, m, n; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray, N, M, O, P, Q, R, S, T; i, j, k, l, m, n, o; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_ref, N, M, O, P, Q, R, S, T; i, j, k, l, m, n, o; bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64);
impl_type_num!(ndarray_source_target, f32, N; ; Complex32);
impl_type_num!(ndarray_source_target, f64, N; ; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M; i; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M; i; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O; i, j; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O; i, j; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O, P; i, j, k; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O, P; i, j, k; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O, P, Q; i, j, k, l; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O, P, Q; i, j, k, l; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O, P, Q, R; i, j, k, l, m; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O, P, Q, R; i, j, k, l, m; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O, P, Q, R, S; i, j, k, l, m, n; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O, P, Q, R, S; i, j, k, l, m, n; Complex64);
impl_type_num!(ndarray_source_target, f32, N, M, O, P, Q, R, S, T; i, j, k, l, m, n, o; Complex32);
impl_type_num!(ndarray_source_target, f64, N, M, O, P, Q, R, S, T; i, j, k, l, m, n, o; Complex64);
