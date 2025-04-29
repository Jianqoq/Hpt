use hpt_common::error::base::TensorError;
use hpt_types::dtype::TypeCommon;

use crate::Tensor;

use super::template::{adaptive_pooling_template, pooling_template};

use hpt_types::type_promote::NormalOut;

use half::{bf16, f16};

impl Tensor {
    pub fn maxpool2d(
        &self,
        kernels_shape: &[i64],
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Tensor, TensorError> {
        macro_rules! maxpool2d {
            ($dtype:ty) => {{
                type T = $dtype;
                type Vec = <$dtype as TypeCommon>::Vec;
                type O = <$dtype as NormalOut>::Output;
                pooling_template::<T, O>(
                    self,
                    &kernels_shape.into(),
                    steps,
                    padding,
                    dilation,
                    |a: T, b: T| a._max(b),
                    |a: Vec, b: Vec| a._max(b),
                    |a: T| a,
                    |a: Vec| a,
                )
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => maxpool2d!(i8),
            hpt_types::dtype::DType::U8 => maxpool2d!(u8),
            hpt_types::dtype::DType::F32 => maxpool2d!(f32),
            hpt_types::dtype::DType::F16 => maxpool2d!(f16),
            hpt_types::dtype::DType::BF16 => maxpool2d!(bf16),
            _ => unimplemented!(),
        }
    }

    pub fn adaptive_maxpool2d(&self, output_size: [i64; 2]) -> Result<Tensor, TensorError> {
        macro_rules! adaptive_maxpool2d {
            ($dtype:ty) => {{
                type T = $dtype;
                type Vec = <$dtype as TypeCommon>::Vec;
                adaptive_pooling_template(
                    self,
                    output_size,
                    |a: T, b: T| a._max(b),
                    |a: Vec, b: Vec| a._max(b),
                    |a: T, _| a,
                    |a: Vec, _| a,
                )
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => adaptive_maxpool2d!(i8),
            hpt_types::dtype::DType::U8 => adaptive_maxpool2d!(u8),
            hpt_types::dtype::DType::F32 => adaptive_maxpool2d!(f32),
            hpt_types::dtype::DType::F16 => adaptive_maxpool2d!(f16),
            hpt_types::dtype::DType::BF16 => adaptive_maxpool2d!(bf16),
            _ => unimplemented!(),
        }
    }
}
