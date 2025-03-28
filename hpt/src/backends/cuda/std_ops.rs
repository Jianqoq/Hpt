use crate::backend::{Cpu, Cuda};
use crate::backends::cuda::utils::binary::binary_normal::*;
use crate::tensor_base::_Tensor;
use crate::Tensor;
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_cudakernels::{ADD, BITAND, BITOR, BITXOR, DIV, MUL, REM, SHL, SHR, SUB};
use hpt_traits::tensor::CommonBounds;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::Cmp;
use hpt_types::type_promote::Eval;
use hpt_types::type_promote::{BitWiseOut, FloatOutBinary, NormalOut};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType};
// define add, sub, mul, rem for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name        op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Add]    [add]         [ADD];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Sub]    [sub]         [SUB];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Mul]    [mul]         [MUL];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Rem]    [rem]         [REM];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Add]    [add]         [ADD];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Sub]    [sub]         [SUB];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Mul]    [mul]         [MUL];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Rem]    [rem]         [REM];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Add]    [add]         [ADD];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Sub]    [sub]         [SUB];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Mul]    [mul]         [MUL];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Rem]    [rem]         [REM];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Add]    [add]         [ADD];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Sub]    [sub]         [SUB];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Mul]    [mul]         [MUL];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Rem]    [rem]         [REM];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + NormalOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<U>>::Output: Cast<<T as NormalOut<U>>::Output>,
    Scalar<T>: NormalOut<Scalar<U>, Output = Scalar<<T as NormalOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        binary_fn_precompiled(
            &self,
            &rhs,
            stringify!(method_name),
            &op,
            None::<out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE, Al>>,
        )
        .unwrap()
    }
}

// define add, sub, mul, rem for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Add]       [add];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Sub]       [sub];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Mul]       [mul];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Rem]       [rem];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Add]       [add];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Sub]       [sub];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Mul]       [mul];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Rem]       [rem];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Add]       [add];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Sub]       [sub];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Mul]       [mul];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Rem]       [rem];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Add]       [add];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Sub]       [sub];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Mul]       [mul];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Rem]       [rem];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + NormalOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<U>>::Output: Cast<<T as NormalOut<U>>::Output>,
    Scalar<T>: NormalOut<Scalar<U>, Output = Scalar<<T as NormalOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name   method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add];

    [Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub];

    [Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul];

    [Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + NormalOut<rhs_type> + DeviceRepr + CudaType,
    <T as NormalOut<rhs_type>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<rhs_type>>::Output: Cast<<T as NormalOut<rhs_type>>::Output>,
    Scalar<T>: NormalOut<Scalar<rhs_type>, Output = Scalar<<T as NormalOut<rhs_type>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = rhs.into();
        let rhs: Tensor<rhs_type, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + NormalOut<rhs_type_ident> + DeviceRepr + CudaType,
    <T as NormalOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as NormalOut<rhs_type_ident>>::Output: Cast<<T as NormalOut<rhs_type_ident>>::Output>,
    Scalar<T>: NormalOut<
        Scalar<rhs_type_ident>,
        Output = Scalar<<T as NormalOut<rhs_type_ident>>::Output>,
    >,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as NormalOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*rhs).into();
        let rhs: Tensor<rhs_type_ident, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add];

    [Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub];

    [Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul];

    [Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>> for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type: CommonBounds + NormalOut<T> + DeviceRepr + CudaType,
    <lhs_type as NormalOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type as NormalOut<T>>::Output: Cast<<lhs_type as NormalOut<T>>::Output>,
    Scalar<lhs_type>: NormalOut<Scalar<T>, Output = Scalar<<lhs_type as NormalOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = self.into();
        let lhs: Tensor<lhs_type, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>>
    for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type_ident: CommonBounds + NormalOut<T> + DeviceRepr + CudaType,
    <lhs_type_ident as NormalOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type_ident as NormalOut<T>>::Output: Cast<<lhs_type_ident as NormalOut<T>>::Output>,
    Scalar<lhs_type_ident>:
        NormalOut<Scalar<T>, Output = Scalar<<lhs_type_ident as NormalOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type_ident as NormalOut<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*self).into();
        let lhs: Tensor<lhs_type_ident, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type         trait_name       method_name         op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitAnd]    [bitand]        [BITAND];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitOr]     [bitor]         [BITOR];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitXor]    [bitxor]        [BITXOR];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitAnd]    [bitand]        [BITAND];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitOr]     [bitor]         [BITOR];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitXor]    [bitxor]        [BITXOR];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitAnd]    [bitand]        [BITAND];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitOr]     [bitor]         [BITOR];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitXor]    [bitxor]        [BITXOR];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitAnd]    [bitand]        [BITAND];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitOr]     [bitor]         [BITOR];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitXor]    [bitxor]        [BITXOR];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        binary_fn_precompiled(
            &self,
            &rhs,
            stringify!(method_name),
            &op,
            None::<out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>>,
        )
        .unwrap()
    }
}

// define add, sub, mul, rem for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name     method_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitOr]     [bitor];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitOr]     [bitor];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitXor]    [bitxor];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: Cast<<T as BitWiseOut<rhs_type>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<rhs_type>, Output = Scalar<<T as BitWiseOut<rhs_type>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = rhs.into();
        let rhs: Tensor<rhs_type, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name        method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]     [bitor];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]     [bitor];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type_ident> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: Cast<<T as BitWiseOut<rhs_type_ident>>::Output>,
    Scalar<T>: BitWiseOut<
        Scalar<rhs_type_ident>,
        Output = Scalar<<T as BitWiseOut<rhs_type_ident>>::Output>,
    >,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*rhs).into();
        let rhs: Tensor<rhs_type_ident, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name         method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>> for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type: CommonBounds + BitWiseOut<T> + DeviceRepr + CudaType,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type as BitWiseOut<T>>::Output: Cast<<lhs_type as BitWiseOut<T>>::Output>,
    Scalar<lhs_type>: BitWiseOut<Scalar<T>, Output = Scalar<<lhs_type as BitWiseOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = self.into();
        let lhs: Tensor<lhs_type, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name        method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>>
    for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type_ident: CommonBounds + BitWiseOut<T> + DeviceRepr + CudaType,
    <lhs_type_ident as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type_ident as BitWiseOut<T>>::Output: Cast<<lhs_type_ident as BitWiseOut<T>>::Output>,
    Scalar<lhs_type_ident>:
        BitWiseOut<Scalar<T>, Output = Scalar<<lhs_type_ident as BitWiseOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type_ident as BitWiseOut<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*self).into();
        let lhs: Tensor<lhs_type_ident, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type         trait_name     method_name  op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Div]    [div]        [DIV];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Div]    [div]        [DIV];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Div]    [div]        [DIV];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Div]    [div]        [DIV];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + FloatOutBinary<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<U>>::Output: Cast<<T as FloatOutBinary<U>>::Output>,
    Scalar<T>: FloatOutBinary<Scalar<U>, Output = Scalar<<T as FloatOutBinary<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        binary_fn_precompiled(
            &self,
            &rhs,
            stringify!(method_name),
            &op,
            None::<out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE, Al>>,
        )
        .unwrap()
    }
}

// define div for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Div]    [div];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Div]    [div];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Div]    [div];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + FloatOutBinary<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<U>>::Output: Cast<<T as FloatOutBinary<U>>::Output>,
    Scalar<T>: FloatOutBinary<Scalar<U>, Output = Scalar<<T as FloatOutBinary<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + FloatOutBinary<rhs_type> + DeviceRepr + CudaType,
    <T as FloatOutBinary<rhs_type>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<rhs_type>>::Output: Cast<<T as FloatOutBinary<rhs_type>>::Output>,
    Scalar<T>:
        FloatOutBinary<Scalar<rhs_type>, Output = Scalar<<T as FloatOutBinary<rhs_type>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as FloatOutBinary<rhs_type>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = rhs.into();
        let rhs: Tensor<rhs_type, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + FloatOutBinary<rhs_type_ident> + DeviceRepr + CudaType,
    <T as FloatOutBinary<rhs_type_ident>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as FloatOutBinary<rhs_type_ident>>::Output:
        Cast<<T as FloatOutBinary<rhs_type_ident>>::Output>,
    Scalar<T>: FloatOutBinary<
        Scalar<rhs_type_ident>,
        Output = Scalar<<T as FloatOutBinary<rhs_type_ident>>::Output>,
    >,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as FloatOutBinary<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*rhs).into();
        let rhs: Tensor<rhs_type_ident, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>> for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type: CommonBounds + FloatOutBinary<T> + DeviceRepr + CudaType,
    <lhs_type as FloatOutBinary<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type as FloatOutBinary<T>>::Output: Cast<<lhs_type as FloatOutBinary<T>>::Output>,
    Scalar<lhs_type>:
        FloatOutBinary<Scalar<T>, Output = Scalar<<lhs_type as FloatOutBinary<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type as FloatOutBinary<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = self.into();
        let lhs: Tensor<lhs_type, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, CUDA_DEVICE, Al>>
    for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type_ident: CommonBounds + FloatOutBinary<T> + DeviceRepr + CudaType,
    <lhs_type_ident as FloatOutBinary<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type_ident as FloatOutBinary<T>>::Output:
        Cast<<lhs_type_ident as FloatOutBinary<T>>::Output>,
    Scalar<lhs_type_ident>:
        FloatOutBinary<Scalar<T>, Output = Scalar<<lhs_type_ident as FloatOutBinary<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type_ident as FloatOutBinary<T>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*self).into();
        let lhs: Tensor<lhs_type_ident, Cuda, CUDA_DEVICE, Al> = lhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        lhs.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type     trait_name         method_name  op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Shl]    [shl]        [SHL];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Shl]    [shl]        [SHL];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Shl]    [shl]        [SHL];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Shl]    [shl]        [SHL];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        binary_fn_precompiled(
            &self,
            &rhs,
            stringify!(method_name),
            &op,
            None::<out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>>,
        )
        .unwrap()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type   out_type    trait_name         method_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Shl]    [shl];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [i8]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [i16]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [i32]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [i64]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [u8]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [u16]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [u32]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [u64]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [f32]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [f64]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Shl]    [shl];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Shl]    [shl];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: Cast<<T as BitWiseOut<rhs_type>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<rhs_type>, Output = Scalar<<T as BitWiseOut<rhs_type>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = rhs.into();
        let rhs: Tensor<rhs_type, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Shl]    [shl];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Shl]    [shl];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Shl]    [shl];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Shl]    [shl];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type_ident> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: Cast<<T as BitWiseOut<rhs_type_ident>>::Output>,
    Scalar<T>: BitWiseOut<
        Scalar<rhs_type_ident>,
        Output = Scalar<<T as BitWiseOut<rhs_type_ident>>::Output>,
    >,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*rhs).into();
        let rhs: Tensor<rhs_type_ident, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shl];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shl];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shl];
)]
impl<T, const DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, DEVICE, Al>> for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type as BitWiseOut<T>>::Output: Cast<<lhs_type as BitWiseOut<T>>::Output>,
    Scalar<lhs_type>: BitWiseOut<Scalar<T>, Output = Scalar<<lhs_type as BitWiseOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cuda, DEVICE, Al>;
    #[track_caller]
    fn shl(self, rhs: rhs_type<T, Cuda, DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = self.into();
        let lhs: Tensor<lhs_type, Cuda, DEVICE, Al> =
            lhs.to_cuda::<DEVICE>().expect("Failed to convert to cuda");
        lhs.inner.as_ref().shl(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type     trait_name         method_name  op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Shr]    [shr]        [SHR];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Shr]    [shr]        [SHR];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Shr]    [shr]        [SHR];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Shr]    [shr]        [SHR];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        binary_fn_precompiled(
            &self,
            &rhs,
            stringify!(method_name),
            &op,
            None::<out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>>,
        )
        .unwrap()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type   out_type    trait_name         method_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Shr]    [shr];
)]
impl<T, U, const CUDA_DEVICE: usize, Al> trait_name<rhs_type<U, Cuda, CUDA_DEVICE, Al>>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr + CudaType,
    U: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<U>>::Output: Cast<<T as BitWiseOut<U>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<U>, Output = Scalar<<T as BitWiseOut<U>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE, Al>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name;
    [Tensor]    [bool]       [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [i8]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [i16]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [i32]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [i64]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [u8]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [u16]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [u32]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [u64]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [f32]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [f64]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Shr]    [shr];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Shr]    [shr];
)]
impl<T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type>>::Output: Cast<<T as BitWiseOut<rhs_type>>::Output>,
    Scalar<T>: BitWiseOut<Scalar<rhs_type>, Output = Scalar<<T as BitWiseOut<rhs_type>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = rhs.into();
        let rhs: Tensor<rhs_type, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Shr]    [shr];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Shr]    [shr];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Shr]    [shr];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Shr]    [shr];
)]
impl<'a, T, const CUDA_DEVICE: usize, Al> trait_name<rhs_type>
    for lhs_type<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + BitWiseOut<rhs_type_ident> + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr + CudaType,
    <T as BitWiseOut<rhs_type_ident>>::Output: Cast<<T as BitWiseOut<rhs_type_ident>>::Output>,
    Scalar<T>: BitWiseOut<
        Scalar<rhs_type_ident>,
        Output = Scalar<<T as BitWiseOut<rhs_type_ident>>::Output>,
    >,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE, Al>;
    #[track_caller]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: Tensor<rhs_type_ident, Cpu, 0, <Al as Allocator>::CpuAllocator> = (*rhs).into();
        let rhs: Tensor<rhs_type_ident, Cuda, CUDA_DEVICE, Al> = rhs
            .to_cuda::<CUDA_DEVICE>()
            .expect("Failed to convert to cuda");
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shr];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shr];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shr];
)]
impl<T, const DEVICE: usize, Al> trait_name<rhs_type<T, Cuda, DEVICE, Al>> for lhs_type
where
    T: CommonBounds + DeviceRepr + CudaType,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr + CudaType,
    <lhs_type as BitWiseOut<T>>::Output: Cast<<lhs_type as BitWiseOut<T>>::Output>,
    Scalar<lhs_type>: BitWiseOut<Scalar<T>, Output = Scalar<<lhs_type as BitWiseOut<T>>::Output>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cuda, DEVICE, Al>;
    #[track_caller]
    fn shr(self, rhs: rhs_type<T, Cuda, DEVICE, Al>) -> Self::Output {
        let lhs: Tensor<lhs_type, Cpu, 0, <Al as Allocator>::CpuAllocator> = self.into();
        let lhs: Tensor<lhs_type, Cuda, DEVICE, Al> =
            lhs.to_cuda::<DEVICE>().expect("Failed to convert to cuda");
        lhs.inner.as_ref().shr(rhs.inner.as_ref()).into()
    }
}

impl<T, const DEVICE: usize, Al> PartialEq<Tensor<T, Cuda, DEVICE, Al>>
    for Tensor<T, Cuda, DEVICE, Al>
where
    T: Eval<Output = bool> + Cmp<Output = bool> + CommonBounds + CudaType + DeviceRepr,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn eq(&self, other: &Tensor<T, Cuda, DEVICE, Al>) -> bool {
        use hpt_traits::tensor::TensorInfo;
        if self.size() != other.size() {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        self.allclose(other, T::ZERO, T::ZERO)
    }
}
