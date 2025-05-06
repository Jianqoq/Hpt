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
            #[cfg(feature = "bool")]
            DType::Bool => display_impl!(bool),
            #[cfg(feature = "i8")]
            DType::I8 => display_impl!(i8),
            #[cfg(feature = "u8")]
            DType::U8 => display_impl!(u8),
            #[cfg(feature = "i16")]
            DType::I16 => display_impl!(i16),
            #[cfg(feature = "u16")]
            DType::U16 => display_impl!(u16),
            #[cfg(feature = "i32")]
            DType::I32 => display_impl!(i32),
            #[cfg(feature = "u32")]
            DType::U32 => display_impl!(u32),
            #[cfg(feature = "i64")]
            DType::I64 => display_impl!(i64),
            #[cfg(feature = "f32")]
            DType::F32 => display_impl!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => display_impl!(half::f16),
            #[cfg(feature = "bf16")]
            DType::BF16 => display_impl!(half::bf16),
            #[cfg(feature = "u64")]
            DType::U64 => display_impl!(u64),
            #[cfg(feature = "f64")]
            DType::F64 => display_impl!(f64),
            _ => unimplemented!("unsupported dtype {:?}", self.dtype),
        }
    }
}
