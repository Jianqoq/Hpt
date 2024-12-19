use std::borrow::Borrow;

use crate::{
    ops::cuda::{cuda_utils::get_module_name_1, unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use tensor_traits::{CommonBounds, FloatUaryOps};
use tensor_types::dtype::Dtype::*;
use tensor_types::{dtype::TypeCommon, into_scalar::IntoScalar, type_promote::FloatOutUnary};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;

fn code_gen<T: TypeCommon, O: TypeCommon>(out: &str, x: &str, op: &str) -> String {
    match T::ID {
        Bool | I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Isize | Usize => match O::ID {
            F16 => {
                format!("{out} = __float2half({op}f(__half2float({x})))",)
            }
            F32 => {
                format!("{out} = {op}f((float){x})")
            }
            F64 => {
                format!("{out} = {op}((double){x})")
            }
            BF16 => {
                unimplemented!("bf16 sin not implemented for cuda")
            }
            _ => unreachable!(),
        },
        F16 => {
            format!("{out} = __float2half({op}f(__half2float({x})))",)
        }
        F32 => {
            format!("{out} = {op}f({x})")
        }
        F64 => {
            format!("{out} = {op}({x})")
        }
        C32 => {
            unimplemented!("c32 sin not implemented for cuda")
        }
        C64 => {
            unimplemented!("c64 sin not implemented for cuda")
        }
        BF16 => {
            unimplemented!("bf16 sin not implemented for cuda")
        }
    }
}

impl<T, const DEVICE_ID: usize> FloatUaryOps for _Tensor<T, Cuda, DEVICE_ID>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds + DeviceRepr,
    FloatUnaryType<T>: CommonBounds + DeviceRepr,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
{
    type Output = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sin", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sin"),
            None::<Self::InplaceOutput>,
        )
    }

    fn cos(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cos", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "cos"),
            None::<Self::InplaceOutput>,
        )
    }

    fn tan(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tan", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "tan"),
            None::<Self::InplaceOutput>,
        )
    }

    fn asin(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asin", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "asin"),
            None::<Self::InplaceOutput>,
        )
    }

    fn acos(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acos", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "acos"),
            None::<Self::InplaceOutput>,
        )
    }

    fn atan(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atan", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "atan"),
            None::<Self::InplaceOutput>,
        )
    }

    fn sinh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sinh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sinh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn cosh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cosh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "cosh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn tanh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tanh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "tanh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn asinh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asinh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "asinh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn acosh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acosh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "acosh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn atanh(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atanh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "atanh"),
            None::<Self::InplaceOutput>,
        )
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<_Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sin", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sin"),
            Some(out),
        )
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cos", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "cos"),
            Some(out),
        )
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tan", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "tan"),
            Some(out),
        )
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asin", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "asin"),
            Some(out),
        )
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acos", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "acos"),
            Some(out),
        )
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atan", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "atan"),
            Some(out),
        )
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sinh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sinh"),
            Some(out),
        )
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cosh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "cosh"),
            Some(out),
        )
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tanh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "tanh"),
            Some(out),
        )
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asinh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "asinh"),
            Some(out),
        )
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acosh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "acosh"),
            Some(out),
        )
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atanh", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "atanh"),
            Some(out),
        )
    }

    fn exp(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "exp"),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "exp"),
            Some(out),
        )
    }

    fn exp2(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp2", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "exp2"),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp2", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "exp2"),
            Some(out),
        )
    }

    fn sqrt(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sqrt", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sqrt"),
            None::<Self::InplaceOutput>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sqrt", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "sqrt"),
            Some(out),
        )
    }

    fn recip(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("recip", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = 1.0f / {x}")
                }
                F64 => {
                    format!("{out} = 1.0 / {x}")
                }
                F16 => {
                    format!("{out} = __float2half(1.0f / __half2float({x}))")
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 recip not implemented for cuda"),
                        F16 => format!("{out} = __float2half(1.0f / __half2float({x}))"),
                        F32 => format!("{out} = 1.0f / (float)({x})"),
                        F64 => format!("{out} = 1.0 / (double)({x})"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 recip not implemented for cuda"),
                C64 => unimplemented!("c64 recip not implemented for cuda"),
                BF16 => unimplemented!("bf16 recip not implemented for cuda"),
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("recip", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = 1.0f / {x}")
                }
                F64 => {
                    format!("{out} = 1.0 / {x}")
                }
                F16 => {
                    format!("{out} = __float2half(1.0f / __half2float({x}))")
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 recip not implemented for cuda"),
                        F16 => format!("{out} = __float2half(1.0f / __half2float({x}))"),
                        F32 => format!("{out} = 1.0f / (float)({x})"),
                        F64 => format!("{out} = 1.0 / (double)({x})"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 recip not implemented for cuda"),
                C64 => unimplemented!("c64 recip not implemented for cuda"),
                BF16 => unimplemented!("bf16 recip not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn ln(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ln", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log"),
            None::<Self::InplaceOutput>,
        )
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ln", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log"),
            Some(out),
        )
    }

    fn log2(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log2", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log2"),
            None::<Self::InplaceOutput>,
        )
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log2", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log2"),
            Some(out),
        )
    }

    fn log10(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log10", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log10"),
            None::<Self::InplaceOutput>,
        )
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log10", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "log10"),
            Some(out),
        )
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("celu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    // CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))
                    format!(
                        "{out} = max({x}, 0.0f) + min(0.0f, {alpha} * (expf({x} / {alpha}) - 1.0f))"
                    )
                }
                F64 => {
                    format!(
                        "{out} = max({x}, 0.0) + min(0.0, {alpha} * (exp({x} / {alpha}) - 1.0))"
                    )
                }
                F16 => {
                    format!(
                        "
                    float alpha_f = __half2float({alpha});
                    float x_f = __half2float({x});
                    {out} = __float2half(max(x_f, 0.0f) + min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                    "
                    )
                }
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 celu not implemented for cuda"),
                        F16 => format!("
                        float alpha_f = __half2float({alpha});
                        float x_f = __half2float({x});
                        {out} = __float2half(max(x_f, 0.0f) + min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                        "),
                        F32 => format!("{out} = (float)max({x}, 0) + min(0.0f, {alpha} * (expf((float){x} / (float){alpha}) - 1.0f))"),
                        F64 => format!("{out} = (double)max({x}, 0) + min(0.0, {alpha} * (exp((double){x} / (double){alpha}) - 1.0))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize => {
                    match FloatUnaryType::<T>::ID {
                        F16 => format!(
                            "
                        float alpha_f = __half2float({alpha});
                        float x_f = __half2float({x});
                        {out} = __float2half(min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                        "
                        ),
                        F32 => format!("{out} = min(0.0f, {alpha} * (expf((float){x} / (float){alpha}) - 1.0f))"),
                        F64 => format!("{out} = min(0.0, {alpha} * (exp((double){x} / (double){alpha}) - 1.0))"),
                        BF16 => unimplemented!("bf16 celu not implemented for cuda"),
                        _ => unreachable!(),
                    }
                }
                Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 celu not implemented for cuda"),
                C64 => unimplemented!("c64 celu not implemented for cuda"),
                BF16 => unimplemented!("bf16 celu not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn celu_<U>(&self, alpha: FloatUnaryType<T>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("celu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    // CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))
                    format!(
                        "{out} = max({x}, 0.0f) + min(0.0f, {alpha} * (expf({x} / {alpha}) - 1.0f))"
                    )
                }
                F64 => {
                    format!(
                        "{out} = max({x}, 0.0) + min(0.0, {alpha} * (exp({x} / {alpha}) - 1.0))"
                    )
                }
                F16 => {
                    format!(
                        "
                    float alpha_f = __half2float({alpha});
                    float x_f = __half2float({x});
                    {out} = __float2half(max(x_f, 0.0f) + min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                    "
                    )
                }
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 celu not implemented for cuda"),
                        F16 => format!("
                        float alpha_f = __half2float({alpha});
                        float x_f = __half2float({x});
                        {out} = __float2half(max(x_f, 0.0f) + min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                        "),
                        F32 => format!("{out} = (float)max({x}, 0) + min(0.0f, {alpha} * (expf((float){x} / (float){alpha}) - 1.0f))"),
                        F64 => format!("{out} = (double)max({x}, 0) + min(0.0, {alpha} * (exp((double){x} / (double){alpha}) - 1.0))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize => {
                    match FloatUnaryType::<T>::ID {
                        F16 => format!(
                            "
                        float alpha_f = __half2float({alpha});
                        float x_f = (float)({x});
                        {out} = __float2half(min(0.0f, alpha_f * (expf(x_f / alpha_f) - 1.0f)))
                        "
                        ),
                        F32 => format!("{out} = min(0.0f, {alpha} * (expf((float){x} / (float){alpha}) - 1.0f))"),
                        F64 => format!("{out} = min(0.0, {alpha} * (exp((double){x} / (double){alpha}) - 1.0))"),
                        BF16 => unimplemented!("bf16 celu not implemented for cuda"),
                        _ => unreachable!(),
                    }
                }
                Bool => {
                    format!("{out} = {x}")
                }
                C32 => unimplemented!("c32 celu not implemented for cuda"),
                C64 => unimplemented!("c64 celu not implemented for cuda"),
                BF16 => unimplemented!("bf16 celu not implemented for cuda"),
            }
            },
            Some(out),
        )
    }

    fn sigmoid(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sigmoid", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = 1.0f / (1.0f + expf(-{x}))")
                }
                F64 => {
                    format!("{out} = 1.0 / (1.0 + exp(-{x}))")
                }
                F16 => {
                    format!("{out} = __float2half(1.0f / (1.0f + expf(-__half2float({x}))))")
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(1.0f / (1.0f + expf(-(float){x})))"),
                        F32 => todo!(),
                        F64 => todo!(),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 sigmoid not implemented for cuda"),
                C64 => unimplemented!("c64 sigmoid not implemented for cuda"),
                BF16 => unimplemented!("bf16 sigmoid not implemented for cuda"),
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sigmoid", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = 1.0f / (1.0f + expf(-{x}))")
                }
                F64 => {
                    format!("{out} = 1.0 / (1.0 + exp(-{x}))")
                }
                F16 => {
                    format!("{out} = __float2half(1.0f / (1.0f + expf(-__half2float({x}))))")
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(1.0f / (1.0f + expf(-(float){x})))"),
                        F32 => todo!(),
                        F64 => todo!(),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 sigmoid not implemented for cuda"),
                C64 => unimplemented!("c64 sigmoid not implemented for cuda"),
                BF16 => unimplemented!("bf16 sigmoid not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("elu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
                    format!("{out} = ({x} > 0.0f) ? {x} : ({alpha} * (expf({x}) - 1.0f))")
                }
                F64 => {
                    format!("{out} = ({x} > 0.0) ? {x} : ({alpha} * (exp({x}) - 1.0))")
                }
                F16 => {
                    format!("
                    float alpha_f = __half2float({alpha});
                    float x_f = __half2float({x});
                    {out} = __float2half((x_f > 0.0f) ? x_f : (alpha_f * (expf(x_f) - 1.0f)))
                    ")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 elu not implemented for cuda"),
                        F16 => format!("                    
                        float alpha_f = (float)({alpha});
                        float x_f = (float)({x});
                        {out} = __float2half((x_f > 0.0f) ? x_f : (alpha_f * (expf(x_f) - 1.0f)))"),
                        F32 => format!("{out} = ((float){x} > 0.0f) ? (float){x} : ({alpha} * (expf((float){x}) - 1.0f))"),
                        F64 => format!("{out} = ((double){x} > 0.0) ? {x} : ({alpha} * (exp((double){x}) - 1.0))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 elu not implemented for cuda"),
                        F16 => format!("{out} = __float2half((float)({x}))"),
                        F32 => format!("{out} = (float){x}"),
                        F64 => format!("{out} = (double){x}"),
                        _ => unreachable!(),
                    }
                }
                BF16 => unimplemented!("bf16 elu not implemented for cuda"),
                C32 => unimplemented!("c32 elu not implemented for cuda"),
                C64 => unimplemented!("c64 elu not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn elu_<U>(&self, alpha: FloatUnaryType<T>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("elu", self),
            |out, x| {
                match T::ID {
            F32 => {
                // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
                format!("{out} = ({x} > 0.0f) ? {x} : ({alpha} * (expf({x}) - 1.0f))")
            }
            F64 => {
                format!("{out} = ({x} > 0.0) ? {x} : ({alpha} * (exp({x}) - 1.0))")
            }
            F16 => {
                format!("
                float alpha_f = __half2float({alpha});
                float x_f = __half2float({x});
                {out} = __float2half((x_f > 0.0f) ? x_f : (alpha_f * (expf(x_f) - 1.0f)))
                ")
            }
            I8 | I16 | I32 | I64 | Isize => {
                match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 elu not implemented for cuda"),
                    F16 => format!("                    
                    float alpha_f = (float)({alpha});
                    float x_f = (float)({x});
                    {out} = __float2half((x_f > 0.0f) ? x_f : (alpha_f * (expf(x_f) - 1.0f)))"),
                    F32 => format!("{out} = ((float){x} > 0.0f) ? (float){x} : ({alpha} * (expf((float){x}) - 1.0f))"),
                    F64 => format!("{out} = ((double){x} > 0.0) ? {x} : ({alpha} * (exp((double){x}) - 1.0))"),
                    _ => unreachable!(),
                }
            }
            U8 | U16 | U32 | U64 | Usize | Bool => {
                match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 elu not implemented for cuda"),
                    F16 => format!("{out} = __float2half((float)({x}))"),
                    F32 => format!("{out} = (float){x}"),
                    F64 => format!("{out} = (double){x}"),
                    _ => unreachable!(),
                }
            }
            BF16 => unimplemented!("bf16 elu not implemented for cuda"),
            C32 => unimplemented!("c32 elu not implemented for cuda"),
            C64 => unimplemented!("c64 elu not implemented for cuda"),
        }
            },
            Some(out),
        )
    }

    fn erf(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("erf", self),
            |out, x| code_gen::<T, FloatUnaryType<T>>(out, x, "erf"),
            None::<Self::InplaceOutput>,
        )
    }

    fn fast_hard_sigmoid(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("fast_hard_sigmoid", self),
            |out, x| {
                match T::ID {
                // 浮点数类型
                F32 => {
                    // fast_hard_sigmoid(x) = max(0, min(1, (x * 0.2 + 0.5)))
                    format!("{out} = fmaxf(0.0f, fminf(1.0f, ({x} * 0.2f + 0.5f)))")
                }
                F64 => {
                    format!("{out} = fmax(0.0, fmin(1.0, ({x} * 0.2 + 0.5)))")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, (__half2float({x}) * 0.2f + 0.5f))))")
                }
                // 整数类型需要先转换为浮点数
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 fast_hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, ((float){x} * 0.2f + 0.5f))))"),
                        F32 => format!("{out} = fmaxf(0.0f, fminf(1.0f, ((float){x} * 0.2f + 0.5f)))"),
                        F64 => format!("{out} = fmax(0.0, fmin(1.0, ((double){x} * 0.2 + 0.5)))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 fast_hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fminf(1.0f, ((float){x} * 0.2f + 0.5f)))"),
                        F32 => format!("{out} = fminf(1.0f, ((float){x} * 0.2f + 0.5f))"),
                        F64 => format!("{out} = fmin(1.0, ((double){x} * 0.2 + 0.5))"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 fast_hard_sigmoid not implemented for cuda"),
                C64 => unimplemented!("c64 fast_hard_sigmoid not implemented for cuda"),
                BF16 => unimplemented!("bf16 fast_hard_sigmoid not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("gelu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    // GELU(x) = x * Φ(x) = x * 0.5 * (1.0 + erf(x/sqrt(2)))
                    format!("{out} = {x} * 0.5f * (1.0f + erff({x} * {}f))", std::f32::consts::FRAC_1_SQRT_2)
                }
                F64 => {
                    format!("{out} = {x} * 0.5 * (1.0 + erf({x} * {}))", std::f64::consts::FRAC_1_SQRT_2)
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * 0.5f * (1.0f + erff(x_f * {}f)))
                        ", std::f32::consts::FRAC_1_SQRT_2
                    )
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 gelu not implemented for cuda"),
                        F16 => format!("{out} = __float2half((float){x} * 0.5f * (1.0f + erff((float){x} * {}f))", std::f32::consts::FRAC_1_SQRT_2),
                        F32 => format!("{out} = (float){x} * 0.5f * (1.0f + erff((float){x} * {}f))", std::f32::consts::FRAC_1_SQRT_2),
                        F64 => format!("{out} = (double){x} * 0.5 * (1.0 + erf((double){x} * {}))", std::f64::consts::FRAC_1_SQRT_2),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 gelu not implemented for cuda"),
                C64 => unimplemented!("c64 gelu not implemented for cuda"),
                BF16 => unimplemented!("bf16 gelu not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("gelu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    // GELU(x) = x * Φ(x) = x * 0.5 * (1.0 + erf(x/sqrt(2)))
                    format!("{out} = {x} * 0.5f * (1.0f + erff({x} * {}f))", std::f32::consts::FRAC_1_SQRT_2)
                }
                F64 => {
                    format!("{out} = {x} * 0.5 * (1.0 + erf({x} * {}))", std::f64::consts::FRAC_1_SQRT_2)
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * 0.5f * (1.0f + erff(x_f * {}f)))
                        ", std::f32::consts::FRAC_1_SQRT_2
                    )
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 gelu not implemented for cuda"),
                        F16 => format!("{out} = __float2half((float){x} * 0.5f * (1.0f + erff((float){x} * {}f))", std::f32::consts::FRAC_1_SQRT_2),
                        F32 => format!("{out} = (float){x} * 0.5f * (1.0f + erff((float){x} * {}f))", std::f32::consts::FRAC_1_SQRT_2),
                        F64 => format!("{out} = (double){x} * 0.5 * (1.0 + erf((double){x} * {}))", std::f64::consts::FRAC_1_SQRT_2),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 gelu not implemented for cuda"),
                C64 => unimplemented!("c64 gelu not implemented for cuda"),
                BF16 => unimplemented!("bf16 gelu not implemented for cuda"),
            }
            },
            Some(out),
        )
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> anyhow::Result<Self::Output>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        let alpha = alpha.into();
        let gamma = gamma.into();
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar());
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("selu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    format!(
                        "{out} = ({x} > 0.0f) ? ({gamma}f * {x}) : ({gamma}f * {alpha}f * (expf({x}) - 1.0f))"
                    )
                }
                F64 => {
                    format!(
                        "{out} = ({x} >= 0.0) ? ({gamma} * {x}) : ({gamma} * {alpha} * (exp({x}) - 1.0))"
                    )
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half((x_f >= 0.0f) ? ({gamma}f * x_f) : ({gamma}f * {alpha}f * (expf(x_f) - 1.0f))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 selu not implemented for cuda"),
                    F16 => format!(
                        "{out} = __float2half((x >= 0) ? ({gamma}f * (float){x}) : ({gamma}f * {alpha}f * (expf((float){x}) - 1.0f)))"
                    ),
                    F32 => format!(
                        "{out} = (x >= 0) ? ({gamma}f * (float){x}) : ({gamma}f * {alpha}f * (expf((float){x}) - 1.0f))"
                    ),
                    F64 => format!(
                        "{out} = (x >= 0) ? ({gamma} * (double){x}) : ({gamma} * {alpha} * (exp((double){x}) - 1.0))"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 selu not implemented for cuda"),
                    F16 => format!("{out} = __float2half({gamma}f * (float){x})"),
                    F32 => format!("{out} = {gamma}f * (float){x}"),
                    F64 => format!("{out} = {gamma}f * (double){x}"),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 selu not implemented for cuda"),
                C64 => unimplemented!("c64 selu not implemented for cuda"),
                BF16 => unimplemented!("bf16 selu not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn selu_<U>(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
        out: U,
    ) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha = alpha.unwrap_or((1.67326319217681884765625).into_scalar());
        let gamma = gamma.unwrap_or((1.05070102214813232421875).into_scalar());
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("selu", self),
            |out, x| {
                match T::ID {
                F32 => {
                    format!(
                        "{out} = ({x} > 0.0f) ? ({gamma}f * {x}) : ({gamma}f * {alpha}f * (expf({x}) - 1.0f))"
                    )
                }
                F64 => {
                    format!(
                        "{out} = ({x} >= 0.0) ? ({gamma} * {x}) : ({gamma} * {alpha} * (exp({x}) - 1.0))"
                    )
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half((x_f >= 0.0f) ? ({gamma}f * x_f) : ({gamma}f * {alpha}f * (expf(x_f) - 1.0f))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 selu not implemented for cuda"),
                    F16 => format!(
                        "{out} = __float2half((x >= 0) ? ({gamma}f * (float){x}) : ({gamma}f * {alpha}f * (expf((float){x}) - 1.0f)))"
                    ),
                    F32 => format!(
                        "{out} = (x >= 0) ? ({gamma}f * (float){x}) : ({gamma}f * {alpha}f * (expf((float){x}) - 1.0f))"
                    ),
                    F64 => format!(
                        "{out} = (x >= 0) ? ({gamma} * (double){x}) : ({gamma} * {alpha} * (exp((double){x}) - 1.0))"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 selu not implemented for cuda"),
                    F16 => format!("{out} = __float2half({gamma}f * (float){x})"),
                    F32 => format!("{out} = {gamma}f * (float){x}"),
                    F64 => format!("{out} = {gamma}f * (double){x}"),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 selu not implemented for cuda"),
                C64 => unimplemented!("c64 selu not implemented for cuda"),
                BF16 => unimplemented!("bf16 selu not implemented for cuda"),
            }
            },
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_sigmoid", self),
            |out, x| {
                match T::ID {
                // 浮点数类型
                F32 => {
                    // hard_sigmoid(x) = max(0, min(1, (x + 3) / 6))
                    format!("{out} = fmaxf(0.0f, fminf(1.0f, ({x} + 3.0f) * 0.1666666666666666666666667f))")
                }
                F64 => {
                    format!("{out} = fmax(0.0, fmin(1.0, ({x} + 3.0) * 0.1666666666666666666666667))")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f)))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f)))"),
                        F32 => format!("{out} = fmaxf(0.0f, fminf(1.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f))"),
                        F64 => format!("{out} = max(0.0, min(1.0, ((double){x} + 3.0) * 0.1666666666666666666666667))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f))"),
                        F32 => format!("{out} = fminf(1.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f)"),
                        F64 => format!("{out} = min(1.0, ((double){x} + 3.0) * 0.1666666666666666666666667)"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 hard_sigmoid not implemented for cuda"),
                C64 => unimplemented!("c64 hard_sigmoid not implemented for cuda"),
                BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_sigmoid", self),
            |out, x| {
                match T::ID {
                // 浮点数类型
                F32 => {
                    // hard_sigmoid(x) = max(0, min(1, (x + 3) / 6))
                    format!("{out} = fmaxf(0.0f, fminf(1.0f, ({x} + 3.0f) * 0.1666666666666666666666667f))")
                }
                F64 => {
                    format!("{out} = fmax(0.0, fmin(1.0, ({x} + 3.0) * 0.1666666666666666666666667))")
                }
                F16 => {
                    format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f)))")
                }
                I8 | I16 | I32 | I64 | Isize => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fmaxf(0.0f, fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f)))"),
                        F32 => format!("{out} = fmaxf(0.0f, fminf(1.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f))"),
                        F64 => format!("{out} = max(0.0, min(1.0, ((double){x} + 3.0) * 0.1666666666666666666666667))"),
                        _ => unreachable!(),
                    }
                }
                U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
                        F16 => format!("{out} = __float2half(fminf(1.0f, (__half2float({x}) + 3.0f) * 0.1666666666666666666666667f))"),
                        F32 => format!("{out} = fminf(1.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f)"),
                        F64 => format!("{out} = min(1.0, ((double){x} + 3.0) * 0.1666666666666666666666667)"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 hard_sigmoid not implemented for cuda"),
                C64 => unimplemented!("c64 hard_sigmoid not implemented for cuda"),
                BF16 => unimplemented!("bf16 hard_sigmoid not implemented for cuda"),
            }
            },
            Some(out),
        )
    }

    fn hard_swish(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_swish", self),
            |out, x| {
                match T::ID {
                // 浮点数类型
                F32 => {
                    // hard_swish(x) = x * min(max(0, x + 3) / 6, 1)
                    format!("{out} = {x} * fminf(fmaxf(0.0f, ({x} + 3.0f) * 0.1666666666666666666666667f), 1.0f)")
                }
                F64 => {
                    format!("{out} = {x} * fmin(fmax(0.0, ({x} + 3.0) * 0.1666666666666666666666667), 1.0)")
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * fminf(fmaxf(0.0f, (x_f + 3.0f) * 0.1666666666666666666666667f), 1.0f))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_swish not implemented for cuda"),
                        F16 => format!("
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * min(max(0.0f, (x_f + 3.0f) * 0.1666666666666666666666667f), 1.0f))"),
                        F32 => format!("{out} = (float){x} * fminf(fmaxf(0.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f), 1.0f)"),
                        F64 => format!("{out} = (double){x} * min(max(0.0, ((double){x} + 3.0) * 0.1666666666666666666666667), 1.0)"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 hard_swish not implemented for cuda"),
                C64 => unimplemented!("c64 hard_swish not implemented for cuda"),
                BF16 => unimplemented!("bf16 hard_swish not implemented for cuda"),
            }
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_swish", self),
            |out, x| {
                match T::ID {
                // 浮点数类型
                F32 => {
                    // hard_swish(x) = x * min(max(0, x + 3) / 6, 1)
                    format!("{out} = {x} * fminf(fmaxf(0.0f, ({x} + 3.0f) * 0.1666666666666666666666667f), 1.0f)")
                }
                F64 => {
                    format!("{out} = {x} * fmin(fmax(0.0, ({x} + 3.0) * 0.1666666666666666666666667), 1.0)")
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * fminf(fmaxf(0.0f, (x_f + 3.0f) * 0.1666666666666666666666667f), 1.0f))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize | U8 | U16 | U32 | U64 | Usize | Bool => {
                    match FloatUnaryType::<T>::ID {
                        BF16 => unimplemented!("bf16 hard_swish not implemented for cuda"),
                        F16 => format!("
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f * min(max(0.0f, (x_f + 3.0f) * 0.1666666666666666666666667f), 1.0f))"),
                        F32 => format!("{out} = (float){x} * fminf(fmaxf(0.0f, ((float){x} + 3.0f) * 0.1666666666666666666666667f), 1.0f)"),
                        F64 => format!("{out} = (double){x} * min(max(0.0, ((double){x} + 3.0) * 0.1666666666666666666666667), 1.0)"),
                        _ => unreachable!(),
                    }
                }
                C32 => unimplemented!("c32 hard_swish not implemented for cuda"),
                C64 => unimplemented!("c64 hard_swish not implemented for cuda"),
                BF16 => unimplemented!("bf16 hard_swish not implemented for cuda"),
            }
            },
            Some(out),
        )
    }

    fn softplus(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softplus", self),
            |out, x| match T::ID {
                // 浮点数类型
                F32 => {
                    format!(
                        "
                        if ({x} > 20.0f) {{
                            {out} = {x};
                        }} else if ({x} < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            {out} = logf(1.0f + expf({x}));
                        }}"
                    )
                }
                F64 => {
                    format!(
                        "
                        if ({x} > 20.0) {{
                            {out} = {x};
                        }} else if ({x} < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            {out} = log(1.0 + exp({x}));
                        }}"
                    )
                }
                F16 => {
                    format!(
                        "
                        float tmp = __half2float({x});
                        if (tmp > 20.0f) {{
                            {out} = {x};
                        }} else if (tmp < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(tmp)));
                        }}"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
                    F16 => format!(
                        "
                        float x_f = (float)({x});
                        if (x_f > 20.0f) {{
                            {out} = __float2half(x_f);
                        }} else if (x_f < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(x_f)));
                        }}
                        "
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            if (x_f > 20.0f) {{
                                {out} = x_f;
                            }} else if (x_f < -20.0f) {{
                                {out} = 0.0f;
                            }} else {{
                                {out} = logf(1.0f + expf(x_f));
                            }}"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            if (x_f > 20.0) {{
                                {out} = x_f;
                            }} else if (x_f < -20.0) {{
                                {out} = 0.0;
                            }} else {{
                                {out} = log(1.0 + exp(x_f));
                            }}"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
                    F16 => format!(
                        "
                        float x_f = (float)({x});
                        if (x_f > 20.0f) {{
                            {out} = __float2half(x_f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(x_f)));
                        }}
                        "
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            if (x_f > 20.0f) {{
                                {out} = x_f;
                            }} else {{
                                {out} = logf(1.0f + expf(x_f));
                            }}"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            if (x_f > 20.0) {{
                                {out} = x_f;
                            }} else {{
                                {out} = log(1.0 + exp(x_f));
                            }}"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 softplus not implemented for cuda"),
                C64 => unimplemented!("c64 softplus not implemented for cuda"),
                BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softplus", self),
            |out, x| match T::ID {
                // 浮点数类型
                F32 => {
                    format!(
                        "
                        if ({x} > 20.0f) {{
                            {out} = {x};
                        }} else if ({x} < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            {out} = logf(1.0f + expf({x}));
                        }}"
                    )
                }
                F64 => {
                    format!(
                        "
                        if ({x} > 20.0) {{
                            {out} = {x};
                        }} else if ({x} < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            {out} = log(1.0 + exp({x}));
                        }}"
                    )
                }
                F16 => {
                    format!(
                        "
                        float tmp = __half2float({x});
                        if (tmp > 20.0f) {{
                            {out} = {x};
                        }} else if (tmp < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(tmp)));
                        }}"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
                    F16 => format!(
                        "
                        float x_f = (float)({x});
                        if (x_f > 20.0f) {{
                            {out} = __float2half(x_f);
                        }} else if (x_f < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(x_f)));
                        }}
                        "
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            if (x_f > 20.0f) {{
                                {out} = x_f;
                            }} else if (x_f < -20.0f) {{
                                {out} = 0.0f;
                            }} else {{
                                {out} = logf(1.0f + expf(x_f));
                            }}"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            if (x_f > 20.0) {{
                                {out} = x_f;
                            }} else if (x_f < -20.0) {{
                                {out} = 0.0;
                            }} else {{
                                {out} = log(1.0 + exp(x_f));
                            }}"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
                    F16 => format!(
                        "
                        float x_f = (float)({x});
                        if (x_f > 20.0f) {{
                            {out} = __float2half(x_f);
                        }} else {{
                            {out} = __float2half(logf(1.0f + expf(x_f)));
                        }}
                        "
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            if (x_f > 20.0f) {{
                                {out} = x_f;
                            }} else {{
                                {out} = logf(1.0f + expf(x_f));
                            }}"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            if (x_f > 20.0) {{
                                {out} = x_f;
                            }} else {{
                                {out} = log(1.0 + exp(x_f));
                            }}"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 softplus not implemented for cuda"),
                C64 => unimplemented!("c64 softplus not implemented for cuda"),
                BF16 => unimplemented!("bf16 softplus not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn softsign(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softsign", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = {x} / (1.0f + fabsf({x}))")
                }
                F64 => {
                    format!("{out} = {x} / (1.0 + fabs({x}))")
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f / (1.0f + fabsf(x_f)))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
                    F16 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = __float2half(x_f / (1.0f + fabsf(x_f)))"
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = x_f / (1.0f + fabsf(x_f))"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            {out} = x_f / (1.0 + fabs(x_f))"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
                    F16 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = __float2half(x_f / (1.0f + x_f))"
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = x_f / (1.0f + x_f)"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            {out} = x_f / (1.0 + x_f)"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 softsign not implemented for cuda"),
                C64 => unimplemented!("c64 softsign not implemented for cuda"),
                BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softsign", self),
            |out, x| match T::ID {
                F32 => {
                    format!("{out} = {x} / (1.0f + fabsf({x}))")
                }
                F64 => {
                    format!("{out} = {x} / (1.0 + fabs({x}))")
                }
                F16 => {
                    format!(
                        "
                        float x_f = __half2float({x});
                        {out} = __float2half(x_f / (1.0f + fabsf(x_f)))"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
                    F16 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = __float2half(x_f / (1.0f + fabsf(x_f)))"
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = x_f / (1.0f + fabsf(x_f))"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            {out} = x_f / (1.0 + fabs(x_f))"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
                    F16 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = __float2half(x_f / (1.0f + x_f))"
                    ),
                    F32 => format!(
                        "
                            float x_f = (float)({x});
                            {out} = x_f / (1.0f + x_f)"
                    ),
                    F64 => format!(
                        "
                            double x_f = (double)({x});
                            {out} = x_f / (1.0 + x_f)"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 softsign not implemented for cuda"),
                C64 => unimplemented!("c64 softsign not implemented for cuda"),
                BF16 => unimplemented!("bf16 softsign not implemented for cuda"),
            },
            Some(out),
        )
    }

    fn mish(&self) -> anyhow::Result<Self::Output> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("mish", self),
            |out, x| match T::ID {
                F32 => {
                    format!(
                        "
                        if ({x} > 20.0f) {{
                            {out} = {x};
                        }} else if ({x} < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = {x} * tanhf(sp);
                        }}"
                    )
                }
                F64 => {
                    format!(
                        "
                        if ({x} > 20.0) {{
                            {out} = {x};
                        }} else if ({x} < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            double sp = log(1.0 + exp({x}));
                            {out} = {x} * tanh(sp);
                        }}"
                    )
                }
                F16 => {
                    format!(
                        "
                        float tmp = __half2float({x});
                        if (tmp > 20.0f) {{
                            {out} = {x};
                        }} else if (tmp < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            float sp = logf(1.0f + expf(tmp));
                            {out} = __float2half(tmp * tanhf(sp));
                        }}"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 mish not implemented for cuda"),
                    F16 => format!(
                        "
                            float tmp = (float)({x});
                            if (tmp > 20.0f) {{
                                {out} = __float2half(tmp);
                            }} else if (tmp < -20.0f) {{
                                {out} = __float2half(0.0f);
                            }} else {{
                                float sp = logf(1.0f + expf(tmp));
                                {out} = __float2half(tmp * tanhf(sp));
                            }}"
                    ),
                    F32 => format!(
                        "
                        float tmp = (float)({x});
                        if (tmp > 20.0f) {{
                            {out} = tmp;
                        }} else if (tmp < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = tmp * tanhf(sp);
                        }}"
                    ),
                    F64 => format!(
                        "
                        double tmp = (double)({x});
                        if (tmp > 20.0) {{
                            {out} = tmp;
                        }} else if (tmp < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            double sp = log(1.0 + exp(tmp));
                            {out} = tmp * tanh(sp);
                        }}"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 mish not implemented for cuda"),
                    F16 => format!(
                        "
                            float tmp = (float)({x});
                            if (tmp > 20.0f) {{
                                {out} = __float2half(tmp);
                            }} else {{
                                float sp = logf(1.0f + expf(tmp));
                                {out} = __float2half(tmp * tanhf(sp));
                            }}"
                    ),
                    F32 => format!(
                        "
                        float tmp = (float)({x});
                        if (tmp > 20.0f) {{
                            {out} = tmp;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = tmp * tanhf(sp);
                        }}"
                    ),
                    F64 => format!(
                        "
                        double tmp = (double)({x});
                        if (tmp > 20.0) {{
                            {out} = tmp;
                        }} else {{
                            double sp = log(1.0 + exp(tmp));
                            {out} = tmp * tanh(sp);
                        }}"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 mish not implemented for cuda"),
                C64 => unimplemented!("c64 mish not implemented for cuda"),
                BF16 => unimplemented!("bf16 mish not implemented for cuda"),
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("mish", self),
            |out, x| match T::ID {
                F32 => {
                    format!(
                        "
                        if ({x} > 20.0f) {{
                            {out} = {x};
                        }} else if ({x} < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = {x} * tanhf(sp);
                        }}"
                    )
                }
                F64 => {
                    format!(
                        "
                        if ({x} > 20.0) {{
                            {out} = {x};
                        }} else if ({x} < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            double sp = log(1.0 + exp({x}));
                            {out} = {x} * tanh(sp);
                        }}"
                    )
                }
                F16 => {
                    format!(
                        "
                        float tmp = __half2float({x});
                        if (tmp > 20.0f) {{
                            {out} = {x};
                        }} else if (tmp < -20.0f) {{
                            {out} = __float2half(0.0f);
                        }} else {{
                            float sp = logf(1.0f + expf(tmp));
                            {out} = __float2half(tmp * tanhf(sp));
                        }}"
                    )
                }
                I8 | I16 | I32 | I64 | Isize => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 mish not implemented for cuda"),
                    F16 => format!(
                        "
                            float tmp = (float)({x});
                            if (tmp > 20.0f) {{
                                {out} = __float2half(tmp);
                            }} else if (tmp < -20.0f) {{
                                {out} = __float2half(0.0f);
                            }} else {{
                                float sp = logf(1.0f + expf(tmp));
                                {out} = __float2half(tmp * tanhf(sp));
                            }}"
                    ),
                    F32 => format!(
                        "
                        float tmp = (float)({x});
                        if (tmp > 20.0f) {{
                            {out} = tmp;
                        }} else if (tmp < -20.0f) {{
                            {out} = 0.0f;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = tmp * tanhf(sp);
                        }}"
                    ),
                    F64 => format!(
                        "
                        double tmp = (double)({x});
                        if (tmp > 20.0) {{
                            {out} = tmp;
                        }} else if (tmp < -20.0) {{
                            {out} = 0.0;
                        }} else {{
                            double sp = log(1.0 + exp(tmp));
                            {out} = tmp * tanh(sp);
                        }}"
                    ),
                    _ => unreachable!(),
                },
                U8 | U16 | U32 | U64 | Usize | Bool => match FloatUnaryType::<T>::ID {
                    BF16 => unimplemented!("bf16 mish not implemented for cuda"),
                    F16 => format!(
                        "
                            float tmp = (float)({x});
                            if (tmp > 20.0f) {{
                                {out} = __float2half(tmp);
                            }} else {{
                                float sp = logf(1.0f + expf(tmp));
                                {out} = __float2half(tmp * tanhf(sp));
                            }}"
                    ),
                    F32 => format!(
                        "
                        float tmp = (float)({x});
                        if (tmp > 20.0f) {{
                            {out} = tmp;
                        }} else {{
                            float sp = logf(1.0f + expf({x}));
                            {out} = tmp * tanhf(sp);
                        }}"
                    ),
                    F64 => format!(
                        "
                        double tmp = (double)({x});
                        if (tmp > 20.0) {{
                            {out} = tmp;
                        }} else {{
                            double sp = log(1.0 + exp(tmp));
                            {out} = tmp * tanh(sp);
                        }}"
                    ),
                    _ => unreachable!(),
                },
                C32 => unimplemented!("c32 mish not implemented for cuda"),
                C64 => unimplemented!("c64 mish not implemented for cuda"),
                BF16 => unimplemented!("bf16 mish not implemented for cuda"),
            },
            Some(out),
        )
    }
}
