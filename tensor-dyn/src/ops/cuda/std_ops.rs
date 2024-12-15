use crate::ops::cuda::binary_normal::*;
use crate::tensor_base::_Tensor;
use crate::Cuda;
use crate::Tensor;
use cudarc::driver::DeviceRepr;
use tensor_traits::tensor::CommonBounds;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::BitWiseOut;
use tensor_types::type_promote::FloatOutBinary;
use tensor_types::type_promote::NormalOut;

// define add, sub, mul, rem for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name   op_string;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Add]    [add]         ["+"];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Add]    [add]         ["+"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Add]    [add]         ["+"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Add]    [add]         ["+"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + NormalOut<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as NormalOut<U>>::Output: CommonBounds + DeviceRepr,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| format!("{} {} {}", x, op_string, y),
            None::<out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE>>,
        )
        .unwrap()
    }
}

// define add, sub, mul, rem for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name   op_string;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + NormalOut<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as NormalOut<U>>::Output: CommonBounds + DeviceRepr,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add]         ["+"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add]         ["+"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type> + DeviceRepr,
    <T as NormalOut<rhs_type>>::Output: CommonBounds + DeviceRepr,
    <T as NormalOut<rhs_type>>::Output: IntoScalar<<T as NormalOut<rhs_type>>::Output>,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define add, sub, mul, rem for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add]         ["+"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add]         ["+"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type_ident> + DeviceRepr,
    <T as NormalOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr,
    <T as NormalOut<rhs_type_ident>>::Output: IntoScalar<<T as NormalOut<rhs_type_ident>>::Output>,
{
    type Output = out_type<<T as NormalOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type_ident, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define add, sub, mul, rem for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add]         ["+"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Add]    [add]         ["+"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type: CommonBounds + NormalOut<T> + DeviceRepr,
    <lhs_type as NormalOut<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type as NormalOut<T>>::Output: IntoScalar<<lhs_type as NormalOut<T>>::Output>,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add]         ["+"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Add]    [add]         ["+"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Add]    [add]         ["+"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Sub]    [sub]         ["-"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Sub]    [sub]         ["-"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Mul]    [mul]         ["*"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Mul]    [mul]         ["*"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Rem]    [rem]         ["%"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Rem]    [rem]         ["%"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type_ident: CommonBounds + NormalOut<T> + DeviceRepr,
    <lhs_type_ident as NormalOut<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type_ident as NormalOut<T>>::Output: IntoScalar<<lhs_type_ident as NormalOut<T>>::Output>,
{
    type Output = out_type<<lhs_type_ident as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type_ident, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type         trait_name       method_name   op_string;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| format!("{} {} {}", x, op_string, y),
            None::<out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE>>,
        )
        .unwrap()
    }
}

// define add, sub, mul, rem for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name   op_string;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitOr]     [bitor]         ["|"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitXor]    [bitxor]        ["^"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + BitWiseOut<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as BitWiseOut<U>>::Output: CommonBounds + DeviceRepr,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define add, sub, mul, rem for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand]        ["&"];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type> + DeviceRepr,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds + DeviceRepr,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define bitwise for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type_ident> + DeviceRepr,
    <T as BitWiseOut<rhs_type_ident>>::Output: CommonBounds + DeviceRepr,
    <T as BitWiseOut<rhs_type_ident>>::Output:
        IntoScalar<<T as BitWiseOut<rhs_type_ident>>::Output>,
{
    type Output = out_type<<T as BitWiseOut<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type_ident, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define bitwise for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type: CommonBounds + BitWiseOut<T> + DeviceRepr,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// define bitwise for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];

    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitOr]    [bitor]         ["|"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitAnd]    [bitand]         ["&"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::BitXor]    [bitxor]         ["^"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type_ident: CommonBounds + BitWiseOut<T> + DeviceRepr,
    <lhs_type_ident as BitWiseOut<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type_ident as BitWiseOut<T>>::Output:
        IntoScalar<<lhs_type_ident as BitWiseOut<T>>::Output>,
{
    type Output = out_type<<lhs_type_ident as BitWiseOut<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type_ident, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// define div for _Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type         trait_name       method_name   op_string;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Div]    [div]        ["/"];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Div]    [div]        ["/"];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Div]    [div]        ["/"];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Div]    [div]        ["/"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + FloatOutBinary<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<U>>::Output: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| format!("{} {} {}", x, op_string, y),
            None::<out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE>>,
        )
        .unwrap()
    }
}

// define div for Tensor
#[duplicate::duplicate_item(
    lhs_type      rhs_type   out_type        trait_name   method_name   op_string;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Div]    [div]         ["/"];
)]
impl<T, U, const CUDA_DEVICE: usize> trait_name<rhs_type<U, Cuda, CUDA_DEVICE>>
    for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + FloatOutBinary<U> + DeviceRepr,
    U: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<U>>::Output: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<U, Cuda, CUDA_DEVICE>) -> Self::Output {
        self.inner.as_ref().method_name(rhs.inner.as_ref()).into()
    }
}

// define div for Tensor and scalar
#[duplicate::duplicate_item(
    lhs_type      rhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div]        ["/"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div]        ["/"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + FloatOutBinary<rhs_type> + DeviceRepr,
    <T as FloatOutBinary<rhs_type>>::Output: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<rhs_type>>::Output: IntoScalar<<T as FloatOutBinary<rhs_type>>::Output>,
{
    type Output = out_type<<T as FloatOutBinary<rhs_type>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define div for Tensor and &scalar
#[duplicate::duplicate_item(
    lhs_type     rhs_type       rhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div]        ["/"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div]        ["/"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div]        ["/"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + FloatOutBinary<rhs_type_ident> + DeviceRepr,
    <T as FloatOutBinary<rhs_type_ident>>::Output: CommonBounds + DeviceRepr,
    <T as FloatOutBinary<rhs_type_ident>>::Output:
        IntoScalar<<T as FloatOutBinary<rhs_type_ident>>::Output>,
{
    type Output = out_type<<T as FloatOutBinary<rhs_type_ident>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type_ident, Cuda, CUDA_DEVICE> = rhs.into();
        self.inner.as_ref().method_name(rhs).into()
    }
}

// define div for scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type   lhs_type     out_type      trait_name     method_name   op_string;
    [Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div]         ["/"];

    [&Tensor]    [bool]       [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [i8]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [i16]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [i32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [i64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [u8]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [u16]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [u32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [u64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [f32]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [f64]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [half::f16]  [Tensor]    [std::ops::Div]    [div]         ["/"];
)]
impl<T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type: CommonBounds + FloatOutBinary<T> + DeviceRepr,
    <lhs_type as FloatOutBinary<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type as FloatOutBinary<T>>::Output: IntoScalar<<lhs_type as FloatOutBinary<T>>::Output>,
{
    type Output = out_type<<lhs_type as FloatOutBinary<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// define div for &scalar and Tensor
#[duplicate::duplicate_item(
    rhs_type     lhs_type       lhs_type_ident   out_type      trait_name     method_name   op_string;
    [Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div]         ["/"];

    [&Tensor]    [&'a bool]        [bool]        [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a i8]          [i8]          [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a i16]         [i16]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a i32]         [i32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a i64]         [i64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a u8]          [u8]          [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a u16]         [u16]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a u32]         [u32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a u64]         [u64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a f32]         [f32]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a f64]         [f64]         [Tensor]    [std::ops::Div]    [div]         ["/"];
    [&Tensor]    [&'a half::f16]   [half::f16]   [Tensor]    [std::ops::Div]    [div]         ["/"];
)]
impl<'a, T, const CUDA_DEVICE: usize> trait_name<rhs_type<T, Cuda, CUDA_DEVICE>> for lhs_type
where
    T: CommonBounds + DeviceRepr,
    lhs_type_ident: CommonBounds + FloatOutBinary<T> + DeviceRepr,
    <lhs_type_ident as FloatOutBinary<T>>::Output: CommonBounds + DeviceRepr,
    <lhs_type_ident as FloatOutBinary<T>>::Output:
        IntoScalar<<lhs_type_ident as FloatOutBinary<T>>::Output>,
{
    type Output = out_type<<lhs_type_ident as FloatOutBinary<T>>::Output, Cuda, CUDA_DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn method_name(self, rhs: rhs_type<T, Cuda, CUDA_DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type_ident, Cuda, CUDA_DEVICE> = self.into();
        lhs.method_name(rhs.inner.as_ref()).into()
    }
}

// impl<T, U, const CUDA_DEVICE: usize> PartialEq<_Tensor<U, Cuda, CUDA_DEVICE>>
//     for _Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds + Convertor,
//     U: CommonBounds + Convertor,
// {
//     fn eq(&self, other: &_Tensor<U, Cuda, CUDA_DEVICE>) -> bool {
//         if self.size() != other.size() {
//             return false;
//         }
//         if self.shape() != other.shape() {
//             return false;
//         }
//         self.allclose(other)
//     }
// }

// macro_rules! normal_scalar_rhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> Add<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//          where T: NormalOut<$type> + CommonBounds,
//          <T as NormalOut<$type>>::Output: CommonBounds,
//          T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn add(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._add(y), |x, y| x._add(y), None::<_Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Mul<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn mul(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._mul(y), |x, y| x._mul(y), None::<_Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Sub<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn sub(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._sub(y), |x, y| x._sub(y), None::<_Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Div<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: FloatOutBinary<$type> + CommonBounds,
//         <T as FloatOutBinary<$type>>::Output: CommonBounds,
//         T::Vec: FloatOutBinary<<$type as TypeCommon>::Vec, Output = <<T as FloatOutBinary<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as FloatOutBinary<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn div(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._div(y), |x, y| x._div(y), None::<_Tensor<<T as FloatOutBinary<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Rem<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn rem(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._rem(y), |x, y| x._rem(y), None::<_Tensor<<T as NormalOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         })*
//     };
// }

// macro_rules! bitwise_scalar_rhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> BitAnd<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//          where T: BitWiseOut<$type> + CommonBounds,
//          <T as BitWiseOut<$type>>::Output: CommonBounds,
//          <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//          T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitand(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._bitand(y), |x, y| x._bitand(y), None::<_Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitOr<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitor(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._bitor(y), |x, y| x._bitor(y), None::<_Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitXor<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitxor(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._bitxor(y), |x, y| x._bitxor(y), None::<_Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shl<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shl(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._shl(y), |x, y| x._shl(y), None::<_Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shr<$type> for $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shr(self, rhs: $type) -> Self::Output {
//                 let rhs: _Tensor<$type, Cuda, CUDA_DEVICE> = rhs.into();
//                 binary_fn_with_out_simd(&self, &rhs, |x, y| x._shr(y), |x, y| x._shr(y), None::<_Tensor<<T as BitWiseOut<$type>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         })*
//     };
// }

// macro_rules! normal_scalar_lhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> Add<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn add(self, rhs: $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>) -> Self::Output {
//                 let lhs: _Tensor<$type, Cuda, CUDA_DEVICE> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._add(y), |x, y| x._add(y), None::<_Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Mul<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn mul(self, rhs: $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>) -> Self::Output {
//                 let lhs: _Tensor<$type, Cuda, CUDA_DEVICE> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._mul(y), |x, y| x._mul(y), None::<_Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Sub<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn sub(self, rhs: $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>) -> Self::Output {
//                 let lhs: _Tensor<$type, Cuda, CUDA_DEVICE> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._sub(y), |x, y| x._sub(y), None::<_Tensor<<$type as NormalOut<T>>::Output, Cuda, CUDA_DEVICE>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Div<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: FloatOutBinary<T> + CommonBounds,
//         <$type as FloatOutBinary<T>>::Output: CommonBounds, $type: FloatOutBinary<T>,
//         <$type as TypeCommon>::Vec: FloatOutBinary<<T as TypeCommon>::Vec, Output = <<$type as FloatOutBinary<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as FloatOutBinary<T>>::Output, Cuda, CUDA_DEVICE>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn div(self, rhs: $($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>) -> Self::Output {
//                 let lhs: _Tensor<$type, Cuda, CUDA_DEVICE> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._div(y), |x, y| x._div(y), None::<_Tensor<<$type as FloatOutBinary<T>>::Output>>).unwrap()
//             }
//         })*
//     };
// }

// macro_rules! bitwise_scalar_lhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> BitAnd<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitand(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
//                 let lhs: _Tensor<$type> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._bitand(y), |x, y| x._bitand(y), None::<_Tensor<<$type as BitWiseOut<T>>::Output>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitOr<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitor(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
//                 let lhs: _Tensor<$type> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._bitor(y), |x, y| x._bitor(y), None::<_Tensor<<$type as BitWiseOut<T>>::Output>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitXor<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitxor(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
//                 let lhs: _Tensor<$type> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._bitxor(y), |x, y| x._bitxor(y), None::<_Tensor<<$type as BitWiseOut<T>>::Output>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shl<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shl(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
//                 let lhs: _Tensor<$type> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._shl(y), |x, y| x._shl(y), None::<_Tensor<<$type as BitWiseOut<T>>::Output>>).unwrap()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shr<$($tokens)*_Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = _Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shr(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
//                 let lhs: _Tensor<$type> = self.into();
//                 binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._shr(y), |x, y| x._shr(y), None::<_Tensor<<$type as BitWiseOut<T>>::Output>>).unwrap()
//             }
//         }
//     )*
//     };
// }

// // impl<T, const CUDA_DEVICE: usize> Not for _Tensor<T, Cuda, CUDA_DEVICE>
// // where
// //     T: BitWiseOut<T> + CommonBounds,
// //     <T as BitWiseOut>::Output: CommonBounds,
// //     T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
// // {
// //     type Output = _Tensor<<T as BitWiseOut<T>>::Output>;
// //     #[cfg_attr(feature = "track_caller", track_caller)]
// //     fn not(self) -> Self::Output {
// //         let lhs: _Tensor<T> = self.into();
// //         uary_fn_with_out_simd(
// //             &lhs,
// //             |x| x._not(),
// //             |x| x._not(),
// //             None::<_Tensor<<T as BitWiseOut<T>>::Output>>,
// //         )
// //         .unwrap()
// //     }
// // }

// // impl<T, const CUDA_DEVICE: usize> Not for &_Tensor<T, Cuda, CUDA_DEVICE>
// // where
// //     T: BitWiseOut<T> + CommonBounds,
// //     <T as BitWiseOut>::Output: CommonBounds,
// //     T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
// // {
// //     type Output = _Tensor<<T as BitWiseOut<T>>::Output>;
// //     #[cfg_attr(feature = "track_caller", track_caller)]
// //     fn not(self) -> Self::Output {
// //         let lhs: _Tensor<T> = self.into();
// //         uary_fn_with_out_simd(
// //             &lhs,
// //             |x| x._not(),
// //             |x| x._not(),
// //             None::<_Tensor<<T as BitWiseOut<T>>::Output>>,
// //         )
// //         .unwrap()
// //     }
// // }

// impl<T, const CUDA_DEVICE: usize> Neg for _Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds,
//     T::Vec: NormalOutUnary<Base = NormalType<T>>,
//     T: NormalOutUnary<Base = NormalType<T>>,
//     _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
// {
//     type Output = _Tensor<NormalType<T>>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn neg(self) -> Self::Output {
//         <_Tensor<T> as NormalUaryOps>::neg(&self).unwrap()
//     }
// }

// impl<T, const CUDA_DEVICE: usize> Neg for &_Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds,
//     T::Vec: NormalOutUnary<Base = NormalType<T>>,
//     T: NormalOutUnary<Base = NormalType<T>>,
//     _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
// {
//     type Output = _Tensor<NormalType<T>>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn neg(self) -> Self::Output {
//         <_Tensor<T> as NormalUaryOps>::neg(&self).unwrap()
//     }
// }

// use half::bf16;
// use half::f16;
// use num::complex::Complex32;
// use num::complex::Complex64;
// normal_scalar_rhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []],
//     [f16, []],
//     [f32, []],
//     [f64, []],
//     [bf16, []],
//     [Complex32, []],
//     [Complex64, []]
// );

// bitwise_scalar_rhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []]
// );

// normal_scalar_rhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// bitwise_scalar_rhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// normal_scalar_lhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []],
//     [f16, []],
//     [f32, []],
//     [f64, []],
//     [bf16, []],
//     [Complex32, []],
//     [Complex64, []]
// );

// bitwise_scalar_lhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []]
// );

// normal_scalar_lhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// bitwise_scalar_lhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// macro_rules! normal_promote_ops_1 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//             where
//             T: CommonBounds + NormalOut<U>,
//             U: CommonBounds,
//             <T as NormalOut<U>>::Output: CommonBounds,
//             <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
//             T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
//         {
//             type Output = Tensor<<T as NormalOut<U>>::Output>;

//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                 self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//             }
//         }
//         )*
//     };
// }

// macro_rules! normal_promote_ops_2 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + NormalOut<U>,
//                 U: CommonBounds,
//                 <T as NormalOut<U>>::Output: CommonBounds,
//                 <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
//                 T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as NormalOut<U>>::Output>;

//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! normal_promote_ops_3 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + NormalOut<U>,
//                 U: CommonBounds,
//                 <T as NormalOut<U>>::Output: CommonBounds,
//                 <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
//                 T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as NormalOut<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! normal_promote_ops_4 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + NormalOut<U>,
//                 U: CommonBounds,
//                 <T as NormalOut<U>>::Output: CommonBounds,
//                 <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
//                 T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as NormalOut<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! normal_promote_ops_assign {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(impl<T, const CUDA_DEVICE: usize> $op<Tensor<T, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE> where T: CommonBounds + NormalOut<Output=T> {
//             fn $op2(&mut self, rhs: Tensor<T, Cuda, CUDA_DEVICE>) {
//                 self.inner.as_ref().$op2(rhs.inner.as_ref())
//             }
//         })*

//         $(impl<'a, T, const CUDA_DEVICE: usize> $op<Tensor<T, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE> where T: CommonBounds + NormalOut<Output=T> {
//             fn $op2(&mut self, rhs: Tensor<T, Cuda, CUDA_DEVICE>) {
//                 self.inner.as_ref().$op2(rhs.inner.as_ref())
//             }
//         })*

//         $(impl<'a, T, const CUDA_DEVICE: usize> $op<&'a Tensor<T, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE> where T: CommonBounds + NormalOut<Output=T> {
//             fn $op2(&mut self, rhs: &'a Tensor<T, Cuda, CUDA_DEVICE>) {
//                 self.inner.as_ref().$op2(rhs.inner.as_ref())
//             }
//         })*

//         $(impl<'a, T, const CUDA_DEVICE: usize> $op<&'a Tensor<T, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE> where T: CommonBounds + NormalOut<Output=T> {
//             fn $op2(&mut self, rhs: &'a Tensor<T, Cuda, CUDA_DEVICE>) {
//                 self.inner.as_ref().$op2(rhs.inner.as_ref())
//             }
//         })*
//     };
// }

// normal_promote_ops_1!(
//     [Add, add, add],
//     [Sub, sub, sub],
//     [Mul, mul, mul],
//     [Rem, rem, rem]
// );

// normal_promote_ops_2!(
//     [Add, add, add],
//     [Sub, sub, sub],
//     [Mul, mul, mul],
//     [Rem, rem, rem]
// );

// normal_promote_ops_3!(
//     [Add, add, add],
//     [Sub, sub, sub],
//     [Mul, mul, mul],
//     [Rem, rem, rem]
// );

// normal_promote_ops_4!(
//     [Add, add, add],
//     [Sub, sub, sub],
//     [Mul, mul, mul],
//     [Rem, rem, rem]
// );

// normal_promote_ops_assign!(
//     [AddAssign, add_assign, add],
//     [SubAssign, sub_assign, sub],
//     [MulAssign, mul_assign, mul],
//     [RemAssign, rem_assign, rem]
// );

// macro_rules! bitwise_promote_ops_1 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! bitwise_promote_ops_2 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! bitwise_promote_ops_3 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! bitwise_promote_ops_4 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// bitwise_promote_ops_1!(
//     [BitAnd, bitand, bitand],
//     [BitOr, bitor, bitor],
//     [BitXor, bitxor, bitxor]
// );

// bitwise_promote_ops_2!(
//     [BitAnd, bitand, bitand],
//     [BitOr, bitor, bitor],
//     [BitXor, bitxor, bitxor]
// );

// bitwise_promote_ops_3!(
//     [BitAnd, bitand, bitand],
//     [BitOr, bitor, bitor],
//     [BitXor, bitxor, bitxor]
// );

// bitwise_promote_ops_4!(
//     [BitAnd, bitand, bitand],
//     [BitOr, bitor, bitor],
//     [BitXor, bitxor, bitxor]
// );

// macro_rules! shift_promote_ops_1 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;

//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! shift_promote_ops_2 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;

//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! shift_promote_ops_3 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;

//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! shift_promote_ops_4 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + BitWiseOut<U>,
//                 U: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: CommonBounds,
//                 <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
//                 T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as BitWiseOut<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// shift_promote_ops_1!([Shl, shl, shl], [Shr, shr, shr]);
// shift_promote_ops_2!([Shl, shl, shl], [Shr, shr, shr]);
// shift_promote_ops_3!([Shl, shl, shl], [Shr, shr, shr]);
// shift_promote_ops_4!([Shl, shl, shl], [Shr, shr, shr]);

// macro_rules! float_binary_promote_ops_1 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + FloatOutBinary<U>,
//                 U: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
//                 T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as FloatOutBinary<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! float_binary_promote_ops_2 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + FloatOutBinary<U>,
//                 U: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
//                 T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as FloatOutBinary<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! float_binary_promote_ops_3 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<&'a Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + FloatOutBinary<U>,
//                 U: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
//                 T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as FloatOutBinary<U>>::Output>;
//                 fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// macro_rules! float_binary_promote_ops_4 {
//     ($([$op:ident, $op2:ident, $op3:ident]),*) => {
//         $(
//             impl<'a, T, U, const CUDA_DEVICE: usize> $op<Tensor<U, Cuda, CUDA_DEVICE>> for &'a Tensor<T, Cuda, CUDA_DEVICE>
//                 where
//                 T: CommonBounds + FloatOutBinary<U>,
//                 U: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: CommonBounds,
//                 <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
//                 T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
//             {
//                 type Output = Tensor<<T as FloatOutBinary<U>>::Output>;
//                 #[cfg_attr(feature = "track_caller", track_caller)]
//                 fn $op2(self, rhs: Tensor<U>) -> Self::Output {
//                     self.inner.as_ref().$op3(rhs.inner.as_ref()).into()
//                 }
//             }
//         )*
//     };
// }

// float_binary_promote_ops_1!([Div, div, div]);
// float_binary_promote_ops_2!([Div, div, div]);
// float_binary_promote_ops_3!([Div, div, div]);
// float_binary_promote_ops_4!([Div, div, div]);

// impl<T, U, const CUDA_DEVICE: usize> PartialEq<Tensor<U, Cuda, CUDA_DEVICE>> for Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds + Convertor,
//     U: CommonBounds + Convertor,
// {
//     fn eq(&self, other: &Tensor<U>) -> bool {
//         if self.size() != other.size() {
//             return false;
//         }
//         if self.shape() != other.shape() {
//             return false;
//         }
//         self.allclose(other)
//     }
// }

// macro_rules! normal_scalar_rhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> Add<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//          where T: NormalOut<$type> + CommonBounds,
//          <T as NormalOut<$type>>::Output: CommonBounds,
//          T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as NormalOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn add(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().add(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Mul<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as NormalOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn mul(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().mul(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Sub<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as NormalOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn sub(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().sub(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Div<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: FloatOutBinary<$type> + CommonBounds,
//         <T as FloatOutBinary<$type>>::Output: CommonBounds,
//         T::Vec: FloatOutBinary<<$type as TypeCommon>::Vec, Output = <<T as FloatOutBinary<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as FloatOutBinary<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn div(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().div(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Rem<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: NormalOut<$type> + CommonBounds,
//         <T as NormalOut<$type>>::Output: CommonBounds,
//         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as NormalOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn rem(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().rem(rhs).into()
//             }
//         })*
//     };
// }

// macro_rules! bitwise_scalar_rhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> BitAnd<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//          where T: BitWiseOut<$type> + CommonBounds,
//          <T as BitWiseOut<$type>>::Output: CommonBounds,
//          <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//          T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as BitWiseOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitand(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().bitand(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitOr<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as BitWiseOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitor(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().bitor(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitXor<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as BitWiseOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitxor(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().bitxor(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shl<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as BitWiseOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shl(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().shl(rhs).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shr<$type> for $($tokens)*Tensor<T, Cuda, CUDA_DEVICE>
//         where T: BitWiseOut<$type> + CommonBounds,
//         <T as BitWiseOut<$type>>::Output: CommonBounds,
//         <T as BitWiseOut<$type>>::Output: IntoScalar<<T as BitWiseOut<$type>>::Output>,
//         T::Vec: BitWiseOut<<$type as TypeCommon>::Vec, Output = <<T as BitWiseOut<$type>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<T as BitWiseOut<$type>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shr(self, rhs: $type) -> Self::Output {
//                 self.inner.as_ref().shr(rhs).into()
//             }
//         })*
//     };
// }

// macro_rules! normal_scalar_lhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> Add<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as NormalOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn add(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.add(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Mul<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as NormalOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn mul(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.mul(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Sub<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: CommonBounds,
//         <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
//         <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as NormalOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn sub(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.sub(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Div<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: FloatOutBinary<T> + CommonBounds,
//         <$type as FloatOutBinary<T>>::Output: CommonBounds, $type: FloatOutBinary<T>,
//         <$type as TypeCommon>::Vec: FloatOutBinary<<T as TypeCommon>::Vec, Output = <<$type as FloatOutBinary<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as FloatOutBinary<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn div(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.div(rhs.inner.as_ref()).into()
//             }
//         })*
//     };
// }

// macro_rules! bitwise_scalar_lhs {
//     (
//         $([
//             $type:ident,
//             [$($tokens:tt)*]
//         ]),*
//     ) => {
//         $(impl<T, const CUDA_DEVICE: usize> BitAnd<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitand(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.bitand(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitOr<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitor(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.bitor(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> BitXor<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn bitxor(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.bitxor(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shl<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shl(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.shl(rhs.inner.as_ref()).into()
//             }
//         }
//         impl<T, const CUDA_DEVICE: usize> Shr<$($tokens)*Tensor<T, Cuda, CUDA_DEVICE>> for $type
//         where T: BitWiseOut<T> + CommonBounds,
//         <$type as BitWiseOut<T>>::Output: CommonBounds, $type: BitWiseOut<T>,
//         <$type as TypeCommon>::Vec: BitWiseOut<<T as TypeCommon>::Vec, Output = <<$type as BitWiseOut<T>>::Output as TypeCommon>::Vec>,
//          {
//             type Output = Tensor<<$type as BitWiseOut<T>>::Output>;
//             #[cfg_attr(feature = "track_caller", track_caller)]
//             fn shr(self, rhs: $($tokens)*Tensor<T>) -> Self::Output {
//                 self.shr(rhs.inner.as_ref()).into()
//             }
//         }
//     )*
//     };
// }

// normal_scalar_rhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []],
//     [f16, []],
//     [f32, []],
//     [f64, []],
//     [bf16, []],
//     [Complex32, []],
//     [Complex64, []]
// );

// bitwise_scalar_rhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []]
// );

// normal_scalar_rhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// bitwise_scalar_rhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// normal_scalar_lhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []],
//     [f16, []],
//     [f32, []],
//     [f64, []],
//     [bf16, []],
//     [Complex32, []],
//     [Complex64, []]
// );

// bitwise_scalar_lhs!(
//     [bool, []],
//     [i8, []],
//     [i16, []],
//     [i32, []],
//     [i64, []],
//     [u8, []],
//     [u16, []],
//     [u32, []],
//     [u64, []]
// );

// normal_scalar_lhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// bitwise_scalar_lhs!(
//     [bool, [&]],
//     [i8, [&]],
//     [i16, [&]],
//     [i32, [&]],
//     [i64, [&]],
//     [u8, [&]],
//     [u16, [&]],
//     [u32, [&]],
//     [u64, [&]],
//     [f16, [&]],
//     [f32, [&]],
//     [f64, [&]],
//     [bf16, [&]],
//     [Complex32, [&]],
//     [Complex64, [&]]
// );

// impl<T, const CUDA_DEVICE: usize> Not for Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: BitWiseOut<T> + CommonBounds,
//     <T as BitWiseOut>::Output: CommonBounds,
//     T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
// {
//     type Output = Tensor<<T as BitWiseOut<T>>::Output>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn not(self) -> Self::Output {
//         self.inner.as_ref().not().into()
//     }
// }

// impl<T, const CUDA_DEVICE: usize> Not for &Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: BitWiseOut<T> + CommonBounds,
//     <T as BitWiseOut>::Output: CommonBounds,
//     T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
// {
//     type Output = Tensor<<T as BitWiseOut<T>>::Output>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn not(self) -> Self::Output {
//         self.inner.as_ref().not().into()
//     }
// }

// impl<T, const CUDA_DEVICE: usize> Neg for Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds,
//     T::Vec: NormalOutUnary<Base = NormalType<T>>,
//     T: NormalOutUnary<Base = NormalType<T>>,
//     Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
// {
//     type Output = Tensor<NormalType<T>>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn neg(self) -> Self::Output {
//         <_Tensor<T> as NormalUaryOps>::neg(self.inner.as_ref())
//             .unwrap()
//             .into()
//     }
// }

// impl<T, const CUDA_DEVICE: usize> Neg for &Tensor<T, Cuda, CUDA_DEVICE>
// where
//     T: CommonBounds,
//     T::Vec: NormalOutUnary<Base = NormalType<T>>,
//     T: NormalOutUnary<Base = NormalType<T>>,
//     Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
// {
//     type Output = Tensor<NormalType<T>>;
//     #[cfg_attr(feature = "track_caller", track_caller)]
//     fn neg(self) -> Self::Output {
//         <_Tensor<T> as NormalUaryOps>::neg(self.inner.as_ref())
//             .unwrap()
//             .into()
//     }
// }
