use crate::{
    ops::cuda::{cuda_utils::get_module_name_1, unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use std::borrow::Borrow;
use tensor_traits::{CommonBounds, NormalUaryOps, TensorLike};
use tensor_types::dtype::Dtype::*;
use tensor_types::type_promote::{NormalOut, NormalOutUnary};

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T, const DEVICE_ID: usize> NormalUaryOps for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr,
    T::Vec: NormalOutUnary<Base = NormalType<T>>,
    T: NormalOutUnary<Base = NormalType<T>>,
    _Tensor<NormalType<T>, Cuda, DEVICE_ID>: TensorLike<NormalType<T>>,
{
    type Output = _Tensor<NormalType<T>, Cuda, DEVICE_ID>;

    type InplaceOutput = _Tensor<NormalType<T>, Cuda, DEVICE_ID>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| match T::ID {
                F16 => {
                    format!("{out} = __float2half(floor(__half2float({x})))",)
                }
                F32 | F64 => {
                    format!("{out} = floor({x})",)
                }
                Bool | I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Isize | Usize => {
                    format!("{out} = {x}",)
                }
                C32 => unimplemented!("c32 floor not implemented for cuda"),
                C64 => unimplemented!("c64 floor not implemented for cuda"),
                BF16 => unimplemented!("bf16 floor not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn floor_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("floor", self),
            |out, x| match T::ID {
                F16 => {
                    format!("{out} = __float2half(floor(__float2half({x})))",)
                }
                F32 | F64 => {
                    format!("{out} = floor({x})",)
                }
                Bool | I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Isize | Usize => {
                    format!("{out} = {x}",)
                }
                C32 => unimplemented!("c32 floor not implemented for cuda"),
                C64 => unimplemented!("c64 floor not implemented for cuda"),
                BF16 => unimplemented!("bf16 floor not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn square(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = __float2half(x * x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = {x} * {x}",)
                }
                Bool => format!("{out} = {x}",),
                I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Isize | Usize => {
                    format!("{out} = {x} * {x}",)
                }
                C32 => unimplemented!("c32 square not implemented for cuda"),
                C64 => unimplemented!("c64 square not implemented for cuda"),
                BF16 => unimplemented!("bf16 square not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn square_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("square", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = __float2half(x * x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = {x} * {x}",)
                }
                Bool => format!("{out} = {x}",),
                I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Isize | Usize => {
                    format!("{out} = {x} * {x}",)
                }
                C32 => unimplemented!("c32 square not implemented for cuda"),
                C64 => unimplemented!("c64 square not implemented for cuda"),
                BF16 => unimplemented!("bf16 square not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn abs(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = abs(x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = abs({x})",)
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}",)
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = abs({x})",)
                }
                C32 => unimplemented!("c32 abs not implemented for cuda"),
                C64 => unimplemented!("c64 abs not implemented for cuda"),
                BF16 => unimplemented!("bf16 abs not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn abs_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("abs", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = abs(x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = abs({x})",)
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}",)
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = abs({x})",)
                }
                C32 => unimplemented!("c32 abs not implemented for cuda"),
                C64 => unimplemented!("c64 abs not implemented for cuda"),
                BF16 => unimplemented!("bf16 abs not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn ceil(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = ceil(x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = ceil({x})",)
                }
                U8 | U16 | U32 | U64 | Usize | Bool | I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = {x}",)
                }
                C32 => unimplemented!("c32 ceil not implemented for cuda"),
                C64 => unimplemented!("c64 ceil not implemented for cuda"),
                BF16 => unimplemented!("bf16 ceil not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }
    fn ceil_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ceil", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = ceil(x);
                    ",
                    )
                }
                F32 | F64 => {
                    format!("{out} = ceil({x})",)
                }
                U8 | U16 | U32 | U64 | Usize | Bool | I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = {x}",)
                }
                C32 => unimplemented!("c32 ceil not implemented for cuda"),
                C64 => unimplemented!("c64 ceil not implemented for cuda"),
                BF16 => unimplemented!("bf16 ceil not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn sign(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = __half2float(copysignf((x != 0.0f), x));
                    ",
                    )
                }
                F32 => {
                    format!("{out} = copysignf(({x} != 0.0f), {x})",)
                }
                F64 => {
                    format!("{out} = copysign(({x} != 0.0), {x})",)
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = (({x} != 0) ? (({x} > 0) ? 1 : -1) : 0)")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = ({x} != 0 ? 1 : 0)",)
                }
                C32 => unimplemented!("c32 sign not implemented for cuda"),
                C64 => unimplemented!("c64 sign not implemented for cuda"),
                BF16 => unimplemented!("bf16 sign not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }
    fn sign_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sign", self),
            |out, x| match T::ID {
                F16 => {
                    format!(
                        "
                    float x = __float2half({x});
                    {out} = __half2float(copysignf((x != 0.0f), x));
                    ",
                    )
                }
                F32 => {
                    format!("{out} = copysignf(({x} != 0.0f), {x})",)
                }
                F64 => {
                    format!("{out} = copysign(({x} != 0.0), {x})",)
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = (({x} != 0) ? (({x} > 0) ? 1 : -1) : 0)")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = ({x} != 0 ? 1 : 0)",)
                }
                C32 => unimplemented!("c32 sign not implemented for cuda"),
                C64 => unimplemented!("c64 sign not implemented for cuda"),
                BF16 => unimplemented!("bf16 sign not implemented for cuda"),
            },
            Some(out),
        )
    }
    fn clamp(&self, min: NormalType<T>, max: NormalType<T>) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| match T::ID {
                F16 => {
                    format!("{out} = __float2half(fminf(fmaxf(__half2float({x}), __half2float({min})), __half2float({max})))")
                }
                F32 => {
                    format!("{out} = fminf(fmaxf({x}, {min}), {max})")
                }
                F64 => {
                    format!("{out} = fmin(fmax({x}, {min}), {max})")
                }
                C32 => unimplemented!("c32 clamp not implemented for cuda"),
                C64 => unimplemented!("c64 clamp not implemented for cuda"),
                BF16 => unimplemented!("bf16 clamp not implemented for cuda"),
                I32 | U32 | I64 | U64 | I8 | U8 | I16 | U16 | Isize | Usize => {
                    format!("{out} = min(max({x}, {min}), {max})")
                }
                Bool => {
                    format!("{out} = {x}")
                }
            },
            None::<Self::Output>,
        )
    }
    fn clamp_<U>(&self, min: NormalType<T>, max: NormalType<T>, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("clamp", self),
            |out, x| match T::ID {
                F16 => {
                    format!("{out} = __float2half(fminf(fmaxf(__half2float({x}), __half2float({min})), __half2float({max})))")
                }
                F32 => {
                    format!("{out} = fminf(fmaxf({x}, {min}), {max})")
                }
                F64 => {
                    format!("{out} = fmin(fmax({x}, {min}), {max})")
                }
                C32 => unimplemented!("c32 clamp not implemented for cuda"),
                C64 => unimplemented!("c64 clamp not implemented for cuda"),
                BF16 => unimplemented!("bf16 clamp not implemented for cuda"),
                I32 | U32 | I64 | U64 | I8 | U8 | I16 | U16 | Isize | Usize => {
                    format!("{out} = min(max({x}, {min}), {max})")
                }
                Bool => {
                    format!("{out} = {x}")
                }
            },
            Some(out),
        )
    }
    fn round(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| match T::ID {
                I32 | U32 | I64 | U64 | I8 | U8 | I16 | U16 | Isize | Usize | Bool => {
                    format!("{out} = {x}")
                }
                F32 => {
                    format!("{out} = roundf({x})")
                }
                F64 => {
                    format!("{out} = round({x})")
                }
                F16 => {
                    format!("{out} = __float2half(roundf(__half2float({x})))")
                }
                C32 => unimplemented!("c32 round not implemented for cuda"),
                C64 => unimplemented!("c64 round not implemented for cuda"),
                BF16 => unimplemented!("bf16 round not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }
    fn round_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("round", self),
            |out, x| match T::ID {
                I32 | U32 | I64 | U64 | I8 | U8 | I16 | U16 | Isize | Usize | Bool => {
                    format!("{out} = {x}")
                }
                F32 => {
                    format!("{out} = roundf({x})")
                }
                F64 => {
                    format!("{out} = round({x})")
                }
                F16 => {
                    format!("{out} = __float2half(roundf(__half2float({x})))")
                }
                C32 => unimplemented!("c32 round not implemented for cuda"),
                C64 => unimplemented!("c64 round not implemented for cuda"),
                BF16 => unimplemented!("bf16 round not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn neg(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| match T::ID {
                F32 | F64 => {
                    format!("{out} = -({x})")
                }
                F16 => {
                    format!("{out} = __hneg({x})")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = -({x})")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    unimplemented!("unsigned neg not implemented for cuda")
                }
                C32 => unimplemented!("c32 neg not implemented for cuda"),
                C64 => unimplemented!("c64 neg not implemented for cuda"),
                BF16 => unimplemented!("bf16 neg not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn neg_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("neg", self),
            |out, x| match T::ID {
                F32 | F64 => {
                    format!("{out} = -({x})")
                }
                F16 => {
                    format!("{out} = __hneg({x})")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = -({x})")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    unimplemented!("unsigned neg not implemented for cuda")
                }
                C32 => unimplemented!("c32 neg not implemented for cuda"),
                C64 => unimplemented!("c64 neg not implemented for cuda"),
                BF16 => unimplemented!("bf16 neg not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn relu(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| match T::ID {
                // 浮点数类型
                F32 => {
                    format!("{out} = fmaxf({x}, 0.0f)")
                }
                F64 => {
                    format!("{out} = fmax({x}, 0.0)")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(__half2float({x}), 0.0f))")
                }
                // 有符号整数类型
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = max({x}, 0)")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 relu not implemented for cuda"),
                C64 => unimplemented!("c64 relu not implemented for cuda"),
                BF16 => unimplemented!("bf16 relu not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn relu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("relu", self),
            |out, x| match T::ID {
                // 浮点数类型
                F32 => {
                    format!("{out} = fmaxf({x}, 0.0f)")
                }
                F64 => {
                    format!("{out} = fmax({x}, 0.0)")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(__half2float({x}), 0.0f))")
                }
                // 有符号整数类型
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = max({x}, 0)")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 relu not implemented for cuda"),
                C64 => unimplemented!("c64 relu not implemented for cuda"),
                BF16 => unimplemented!("bf16 relu not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = fmaxf({x}, {alpha} * {x})")
                }
                F64 => {
                    format!("{out} = fmax({x}, {alpha} * {x})")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(__half2float({x}), __half2float({alpha}) * __half2float({x})))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = ({x} > 0) ? {x} : ({alpha} * {x})")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 leaky_relu not implemented for cuda"),
                C64 => unimplemented!("c64 leaky_relu not implemented for cuda"),
                BF16 => unimplemented!("bf16 leaky_relu not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = fmaxf({x}, {alpha} * {x})")
                }
                F64 => {
                    format!("{out} = fmax({x}, {alpha} * {x})")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(__half2float({x}), __half2float({alpha}) * __half2float({x})))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = ({x} > 0) ? {x} : ({alpha} * {x})")
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 leaky_relu not implemented for cuda"),
                C64 => unimplemented!("c64 leaky_relu not implemented for cuda"),
                BF16 => unimplemented!("bf16 leaky_relu not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = fminf(fmaxf({x}, 0.0f), 6.0f)")
                }
                F64 => {
                    format!("{out} = fmin(fmax({x}, 0.0), 6.0)")
                }
                F16 => {
                    format!("{out} = __float2half(fminf(fmaxf(__half2float({x}), 0.0f), 6.0f))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = min(max({x}, 0), 6)")
                }
                U8 | U16 | U32 | U64 | Usize => {
                    format!("{out} = min({x}, 6)")
                }
                Bool => {
                    format!("{out} = {x}")
                }
                // 其他类型暂不支持
                C32 => unimplemented!("c32 relu6 not implemented for cuda"),
                C64 => unimplemented!("c64 relu6 not implemented for cuda"),
                BF16 => unimplemented!("bf16 relu6 not implemented for cuda"),
            },
            None::<Self::Output>,
        )
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("leaky_relu", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = fminf(fmaxf({x}, 0.0f), 6.0f)")
                }
                F64 => {
                    format!("{out} = fmin(fmax({x}, 0.0), 6.0)")
                }
                F16 => {
                    format!("{out} = __float2half(fminf(fmaxf(__half2float({x}), 0.0f), 6.0f))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    format!("{out} = min(max({x}, 0), 6)")
                }
                U8 | U16 | U32 | U64 | Usize => {
                    format!("{out} = min({x}, 6)")
                }
                Bool => {
                    format!("{out} = {x}")
                }
                // 其他类型暂不支持
                C32 => unimplemented!("c32 relu6 not implemented for cuda"),
                C64 => unimplemented!("c64 relu6 not implemented for cuda"),
                BF16 => unimplemented!("bf16 relu6 not implemented for cuda"),
            },
            Some(out),
        )
    }
}
