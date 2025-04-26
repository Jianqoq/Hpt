use hpt_display::display;

use crate::{DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION};
use crate::{DType, Tensor};
use std::fmt::Display;
use std::sync::atomic::Ordering;

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        macro_rules! display_impl {
            ($cast_type:ty) => {
                display(
                    self.data.cast::<$cast_type>(),
                    self.layout.shape().as_slice(),
                    self.layout.strides().as_slice(),
                    f,
                    lr_element_size,
                    precision,
                    false,
                )
            };
        }
        match self.dtype {
            DType::Bool => {
                display_impl!(bool)
            }
            DType::I8 => {
                display_impl!(i8)
            }
            DType::U8 => {
                display_impl!(u8)
            }
            DType::I16 => {
                display_impl!(i16)
            }
            DType::U16 => {
                display_impl!(u16)
            }
            DType::I32 => {
                display_impl!(i32)
            }
            DType::U32 => {
                display_impl!(u32)
            }
            DType::I64 => {
                display_impl!(i64)
            }
            DType::F32 => {
                display_impl!(f32)
            }
            DType::F16 => {
                display_impl!(half::f16)
            }
            DType::BF16 => {
                display_impl!(half::bf16)
            }
        }
    }
}
