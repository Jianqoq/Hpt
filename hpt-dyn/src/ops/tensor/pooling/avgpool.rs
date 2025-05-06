use hpt_common::error::base::TensorError;
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::FloatOutBinary};

use crate::Tensor;

use super::template::{adaptive_pooling_template, pooling_template};

use hpt_types::traits::VecTrait;

use hpt_types::type_promote::NormalOut;


impl Tensor {
    pub fn avgpool2d(
        &self,
        kernels_shape: &[i64],
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Tensor, TensorError> {
        let kernel_size = kernels_shape.iter().product::<i64>();
        macro_rules! avgpool2d {
            ($dtype:ty) => {{
                type T = $dtype;
                type Vec = <$dtype as TypeCommon>::Vec;
                type O = <$dtype as FloatOutBinary>::Output;
                type OVec = <<$dtype as FloatOutBinary>::Output as TypeCommon>::Vec;
                let kernel_size: O = kernel_size.cast();
                let kernel_size_vec = OVec::splat(kernel_size);
                pooling_template::<T, O>(
                    self,
                    &kernels_shape.into(),
                    steps,
                    padding,
                    dilation,
                    |a: T, b: T| a._add(b),
                    |a: Vec, b: Vec| a._add(b),
                    |a: T| a._div(kernel_size),
                    |a: Vec| a._div(kernel_size_vec),
                )
            }};
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => avgpool2d!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => avgpool2d!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => avgpool2d!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => avgpool2d!(half::f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => avgpool2d!(half::bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => avgpool2d!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => avgpool2d!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => avgpool2d!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => avgpool2d!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => avgpool2d!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => avgpool2d!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => avgpool2d!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => avgpool2d!(f64),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Tensor, TensorError> {
        macro_rules! adaptive_avgpool2d {
            ($dtype:ty) => {{
                type T = $dtype;
                type Vec = <$dtype as TypeCommon>::Vec;
                type O = <$dtype as FloatOutBinary>::Output;
                type OVec = <<$dtype as FloatOutBinary>::Output as TypeCommon>::Vec;
                adaptive_pooling_template(
                    self,
                    output_size,
                    |a: T, b: T| a._add(b),
                    |a: Vec, b: Vec| a._add(b),
                    |a: T, kernel_size: O| a._div(kernel_size),
                    |a: Vec, kernel_size_vec: OVec| a._div(kernel_size_vec),
                )
            }};
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => adaptive_avgpool2d!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => adaptive_avgpool2d!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => adaptive_avgpool2d!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => adaptive_avgpool2d!(half::f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => adaptive_avgpool2d!(half::bf16),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => adaptive_avgpool2d!(bool),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => adaptive_avgpool2d!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => adaptive_avgpool2d!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => adaptive_avgpool2d!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => adaptive_avgpool2d!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => adaptive_avgpool2d!(i64),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => adaptive_avgpool2d!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => adaptive_avgpool2d!(f64),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
}
