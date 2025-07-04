use crate::backend::Cpu;
use crate::tensor::{DiffTensor, Tensor};
use crate::tensor_base::_Tensor;
use half::bf16;
use half::f16;
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_allocator::Backend;
use hpt_common::shape::shape::Shape;
use hpt_common::strides::strides_utils::shape_to_strides;
use hpt_common::utils::pointer::Pointer;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::TensorLike;
use num::complex::{Complex32, Complex64};
use std::alloc::Layout;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::Arc;

macro_rules! from_scalar {
    ($($t:ident),*) => {
        $(
            impl<const DEVICE: usize, A> Into<_Tensor<$t, Cpu, DEVICE, A>> for $t where A: Allocator, A::Output: AllocatorOutputRetrive {
                fn into(self) -> _Tensor<$t, Cpu, DEVICE, A> {
                    let mut ret = _Tensor::<$t, Cpu, DEVICE, A>::empty(vec![1]).unwrap();
                    ret.as_raw_mut()[0] = self;
                    return ret;
                }
            }
            impl<const DEVICE: usize, A> Into<Tensor<$t, Cpu, DEVICE, A>> for $t where A: Allocator, A::Output: AllocatorOutputRetrive {
                fn into(self) -> Tensor<$t, Cpu, DEVICE, A> {
                    Tensor {
                        inner: Arc::new(self.into()),
                    }
                }
            }
        )*
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
            impl<const DEVICE: usize, A> From<Vec<$t>> for _Tensor<$t, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
                fn from(data: Vec<$t>) -> Self {
                    let mut ptr = data.as_ptr() as *mut $t;
                    let length = data.len();
                    let res_shape = Shape::from(vec![length as i64]);
                    let layout;
                    let allocator = A::new();
                    if (ptr as usize) % 8 == 0 {
                        let _ = ManuallyDrop::new(data);
                        layout = Layout::from_size_align(length * std::mem::size_of::<$t>(), 8).unwrap();
                        allocator.insert_ptr(ptr as *mut u8, DEVICE);
                    } else {
                        layout = Layout::from_size_align(length * std::mem::size_of::<$t>(), 8).unwrap();
                        let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                        ptr = allocate_res.get_ptr() as *mut $t;
                        unsafe {
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, length);
                        }
                    }
                    let ly = hpt_common::layout::layout::Layout::new(res_shape, vec![1]);
                    return _Tensor {
                        data: Pointer::new(ptr, length as i64),
                        parent: None,
                        layout: ly,
                        mem_layout: Arc::new(layout),
                        backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                        phantom: PhantomData,
                    };
                }
            }
            impl<const DEVICE: usize> From<Vec<$t>> for Tensor<$t, Cpu, DEVICE> {
                fn from(data: Vec<$t>) -> Self {
                    Tensor {
                        inner: Arc::new(data.into()),
                    }
                }
            }
        )*
    };
    (ndarray, $($generic:ident),*; $($vars:ident),*; $ct:ident, $($t:ident),*) => {
            impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
                fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                    let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                    let shape = Shape::from(vec![$($generic as i64), *]);

                    repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element));
                    let mut ptr = vec.as_mut_ptr();
                    let length = repeate_generic!(mul, $($generic), *);
                    let layout;
                    let allocator = A::new();
                    if (ptr as usize) % 8 == 0 {
                        let _ = ManuallyDrop::new(vec);
                        layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                        allocator.insert_ptr(ptr as *mut u8, DEVICE);
                    } else {
                        layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                        let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                        ptr = allocate_res.get_ptr() as *mut $ct;
                        unsafe {
                            std::ptr::copy_nonoverlapping(vec.as_ptr(), ptr, vec.len());
                        }
                    }
                    let strides = shape_to_strides(&shape);
                    let ly = hpt_common::layout::layout::Layout::new(shape, strides);
                    return _Tensor {
                        data: Pointer::new(ptr, length as i64),
                        parent: None,
                        layout: ly,
                        mem_layout: Arc::new(layout),
                        backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                        phantom: PhantomData,
                    };
                }
            }
            impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
                fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                    Tensor {
                        inner: Arc::new(data.into()),
                    }
                }
            }
            impl_type_num!(ndarray, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                let allocator = A::new();
                if (ptr as usize) % 8 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    allocator.insert_ptr(ptr as *mut u8, DEVICE);
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                    ptr = allocate_res.get_ptr() as *mut $ct;
                    unsafe {
                        std::ptr::copy_nonoverlapping(vec.as_ptr(), ptr, vec.len());
                    }
                }
                let strides = shape_to_strides(&shape);

                let ly = hpt_common::layout::layout::Layout::new(shape, strides);
                return _Tensor {
                    data: Pointer::new(ptr, length as i64),
                    parent: None,
                    layout: ly,
                    mem_layout: Arc::new(layout),
                    backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                    phantom: PhantomData,
                };
            }
        }
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $ct)> for Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                Tensor {
                    inner: Arc::new(data.into()),
                }
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
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $source)> for Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: repeate_generic!(nested_array_type, $($generic), *; $source)) -> Self {
                Tensor {
                    inner: Arc::new(data.into()),
                }
            }
        }
        impl_type_num!(ndarray_source_target, $source, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray_source_target, $source:ident, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
    impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $source)> for _Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
        fn from(data: repeate_generic!(nested_array_type, $($generic), *; $source)) -> Self {
            let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
            let shape = Shape::from(vec![$($generic as i64), *]);

            repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(element.into()));
            let mut ptr = vec.as_mut_ptr();
            let length = repeate_generic!(mul, $($generic), *);
            let layout;
            let allocator = A::new();
            if (ptr as usize) % 8 == 0 {
                let _ = ManuallyDrop::new(vec);
                layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                allocator.insert_ptr(ptr as *mut u8, DEVICE);
            } else {
                layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                ptr = allocate_res.get_ptr() as *mut $ct;
                unsafe {
                    std::ptr::copy_nonoverlapping(vec.as_ptr(), ptr, vec.len());
                }
            }
            let strides = shape_to_strides(&shape);

            let ly = hpt_common::layout::layout::Layout::new(shape, strides);
            return _Tensor {
                data: Pointer::new(ptr, length as i64),
                parent: None,
                layout: ly,
                mem_layout: Arc::new(layout),
                backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                phantom: PhantomData,
            };
        }
    }
    impl<$(const $generic: usize), *, const DEVICE: usize, A> From<repeate_generic!(nested_array_type, $($generic), *; $source)> for Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
        fn from(data: repeate_generic!(nested_array_type, $($generic), *; $source)) -> Self {
            Tensor {
                    inner: Arc::new(data.into()),
                }
            }
        }
    };

    (ndarray_ref, $($generic:ident),*; $($vars:ident),*; $ct:ident, $($t:ident),*) => {
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(*element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                let allocator = A::new();
                if (ptr as usize) % 8 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    allocator.insert_ptr(ptr as *mut u8, DEVICE);
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                    ptr = allocate_res.get_ptr() as *mut $ct;
                    unsafe {
                        std::ptr::copy_nonoverlapping(vec.as_ptr(), ptr, vec.len());
                    }
                }
                let strides = shape_to_strides(&shape);

                let ly = hpt_common::layout::layout::Layout::new(shape, strides);
                return _Tensor {
                    data: Pointer::new(ptr, length as i64),
                    parent: None,
                    layout: ly,
                    mem_layout: Arc::new(layout),
                    backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                    phantom: PhantomData,
                };
            }
        }
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                Tensor {
                    inner: Arc::new(data.into()),
                }
            }
        }
        impl_type_num!(ndarray_ref, $($generic), *; $($vars), *; $($t),*);
    };
    (ndarray_ref, $($generic:ident),*; $($vars:ident),*; $ct:ident) => {
        impl<$(const $generic: usize), *, const DEVICE: usize, A> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for _Tensor<$ct, Cpu, DEVICE, A> where A: Allocator, A::Output: AllocatorOutputRetrive {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                let mut vec: Vec<$ct> = Vec::with_capacity(repeate_generic!(operations, *, $($generic), *));
                let shape = Shape::from(vec![$($generic as i64), *]);

                repeate_generic!(iterate, data; vec; $($vars), *).for_each(|element| vec.push(*element));
                let mut ptr = vec.as_mut_ptr();
                let length = repeate_generic!(mul, $($generic), *);
                let layout;
                let allocator = A::new();
                if (ptr as usize) % 8 == 0 {
                    let _ = ManuallyDrop::new(vec);
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    allocator.insert_ptr(ptr as *mut u8, DEVICE);
                } else {
                    layout = Layout::from_size_align(length * std::mem::size_of::<$ct>(), 8).unwrap();
                    let allocate_res = allocator.allocate(layout, DEVICE).unwrap();
                    ptr = allocate_res.get_ptr() as *mut $ct;
                    unsafe {
                        std::ptr::copy_nonoverlapping(vec.as_ptr(), ptr, vec.len());
                    }
                }
                let strides = shape_to_strides(&shape);

                let ly = hpt_common::layout::layout::Layout::new(shape, strides);
                return _Tensor {
                    data: Pointer::new(ptr, length as i64),
                    parent: None,
                    layout: ly,
                    mem_layout: Arc::new(layout),
                    backend: Backend::<Cpu>::new(ptr as u64, DEVICE, true),
                    phantom: PhantomData,
                };
            }
        }
        impl<$(const $generic: usize), *, const DEVICE: usize> From<&repeate_generic!(nested_array_type, $($generic), *; $ct)> for Tensor<$ct, Cpu, DEVICE> {
            fn from(data: &repeate_generic!(nested_array_type, $($generic), *; $ct)) -> Self {
                Tensor {
                    inner: Arc::new(data.into()),
                }
            }
        }
    };
}

/// This macro is used to generate the nested array type
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
impl_type_num!(
    vec, bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, Complex32, Complex64
); // prettier-ignore
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

impl<T, const DEVICE: usize, Al> Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
{
    /// Creates a new tensor from the provided data.
    pub fn new<A>(data: A) -> Self
    where
        A: Into<Tensor<T, Cpu, DEVICE, Al>>,
    {
        data.into()
    }
}

impl<T, const DEVICE: usize, Al> DiffTensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
{
    /// Creates a new differentiable tensor from the provided data.
    pub fn new<A>(data: A) -> Self
    where
        A: Into<Tensor<T, Cpu, DEVICE, Al>>,
    {
        let ret = data.into();
        DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        }
    }
}
