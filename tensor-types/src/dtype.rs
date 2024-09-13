use half::{ bf16, f16 };
use tensor_macros::infer_enum_type;
use std::fmt::{ Debug, Display };
use serde::{ Deserialize, Serialize };
use crate::{
    into_vec::IntoVec,
    type_promote::{ BitWiseOut, Eval, FloatOutBinary, FloatOutUnary, NormalOut },
    vectors::traits::{ Init, VecCommon, VecTrait },
};
#[cfg(target_feature = "avx512f")]
use crate::vectors::_512bit::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    BF16,
    F16,
    F32,
    F64,
    C32,
    C64,
    Isize,
    Usize,
}

impl Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Bool => write!(f, "bool"),
            Dtype::I8 => write!(f, "i8"),
            Dtype::U8 => write!(f, "u8"),
            Dtype::I16 => write!(f, "i16"),
            Dtype::U16 => write!(f, "u16"),
            Dtype::I32 => write!(f, "i32"),
            Dtype::U32 => write!(f, "u32"),
            Dtype::I64 => write!(f, "i64"),
            Dtype::U64 => write!(f, "u64"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::F16 => write!(f, "f16"),
            Dtype::F32 => write!(f, "f32"),
            Dtype::F64 => write!(f, "f64"),
            Dtype::C32 => write!(f, "c32"),
            Dtype::C64 => write!(f, "c64"),
            Dtype::Isize => write!(f, "isize"),
            Dtype::Usize => write!(f, "usize"),
        }
    }
}

impl Dtype {
    pub const fn size(&self) -> usize {
        match self {
            Dtype::Bool => std::mem::size_of::<bool>(),
            Dtype::I8 => std::mem::size_of::<i8>(),
            Dtype::U8 => std::mem::size_of::<u8>(),
            Dtype::I16 => std::mem::size_of::<i16>(),
            Dtype::U16 => std::mem::size_of::<u16>(),
            Dtype::I32 => std::mem::size_of::<i32>(),
            Dtype::U32 => std::mem::size_of::<u32>(),
            Dtype::I64 => std::mem::size_of::<i64>(),
            Dtype::U64 => std::mem::size_of::<u64>(),
            Dtype::BF16 => std::mem::size_of::<u16>(),
            Dtype::F16 => std::mem::size_of::<u16>(),
            Dtype::F32 => std::mem::size_of::<f32>(),
            Dtype::F64 => std::mem::size_of::<f64>(),
            Dtype::C32 => std::mem::size_of::<f32>() * 2,
            Dtype::C64 => std::mem::size_of::<f64>() * 2,
            Dtype::Isize => std::mem::size_of::<isize>(),
            Dtype::Usize => std::mem::size_of::<usize>(),
        }
    }
    pub const fn bits(&self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I8 => 8,
            Dtype::U8 => 8,
            Dtype::I16 => 16,
            Dtype::U16 => 16,
            Dtype::I32 => 32,
            Dtype::U32 => 32,
            Dtype::I64 => 64,
            Dtype::U64 => 64,
            Dtype::BF16 => 16,
            Dtype::F16 => 16,
            Dtype::F32 => 32,
            Dtype::F64 => 64,
            Dtype::C32 => 32,
            Dtype::C64 => 64,
            Dtype::Isize => 64,
            Dtype::Usize => 64,
        }
    }
}

pub trait TypeCommon where Self: Sized {
    const ID: Dtype;
    const MAX: Self;
    const MIN: Self;
    const ZERO: Self;
    const ONE: Self;
    const INF: Self;
    const NEG_INF: Self;
    const TWO: Self;
    const STR: &'static str;
    type Vec: VecTrait<Self> +
        Init<Self> +
        VecCommon +
        Send +
        Copy +
        IntoVec<Self::Vec> +
        Sync +
        Debug;
}

macro_rules! impl_type_common {
    (
        $type:ty,
        $dtype:ident,
        $max:expr,
        $min:expr,
        $zero:expr,
        $one:expr,
        $inf:expr,
        $neg_inf:expr,
        $two:expr,
        $str:expr,
        $vec:ty
    ) => {
        impl TypeCommon for $type {
            const ID: Dtype = Dtype::$dtype;
            const MAX: Self = $max;
            const MIN: Self = $min;
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const INF: Self = $inf;
            const NEG_INF: Self = $neg_inf;
            const TWO: Self = $two;
            const STR: &'static str = $str;
            type Vec = $vec;
        }
    };
}

#[cfg(target_feature = "avx2")]
mod type_impl {
    use crate::vectors::_256bit::*;
    use half::*;
    use num_complex::{ Complex32, Complex64 };
    use super::{ TypeCommon, Dtype };

    use crate::vectors::_256bit::*;
    impl_type_common!(
        bool,
        Bool,
        true,
        false,
        false,
        true,
        true,
        false,
        false,
        "bool",
        boolx32::boolx32
    );
    impl_type_common!(i8, I8, i8::MAX, i8::MIN, 0, 1, i8::MAX, i8::MIN, 2, "i8", i8x32::i8x32);
    impl_type_common!(u8, U8, u8::MAX, u8::MIN, 0, 1, u8::MAX, u8::MIN, 2, "u8", u8x32::u8x32);
    impl_type_common!(
        i16,
        I16,
        i16::MAX,
        i16::MIN,
        0,
        1,
        i16::MAX,
        i16::MIN,
        2,
        "i16",
        i16x16::i16x16
    );
    impl_type_common!(
        u16,
        U16,
        u16::MAX,
        u16::MIN,
        0,
        1,
        u16::MAX,
        u16::MIN,
        2,
        "u16",
        u16x16::u16x16
    );
    impl_type_common!(
        i32,
        I32,
        i32::MAX,
        i32::MIN,
        0,
        1,
        i32::MAX,
        i32::MIN,
        2,
        "i32",
        i32x8::i32x8
    );
    impl_type_common!(
        u32,
        U32,
        u32::MAX,
        u32::MIN,
        0,
        1,
        u32::MAX,
        u32::MIN,
        2,
        "u32",
        u32x8::u32x8
    );
    impl_type_common!(
        i64,
        I64,
        i64::MAX,
        i64::MIN,
        0,
        1,
        i64::MAX,
        i64::MIN,
        2,
        "i64",
        i64x4::i64x4
    );
    impl_type_common!(
        u64,
        U64,
        u64::MAX,
        u64::MIN,
        0,
        1,
        u64::MAX,
        u64::MIN,
        2,
        "u64",
        u64x4::u64x4
    );
    impl_type_common!(
        f32,
        F32,
        f32::MAX,
        f32::MIN,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        "f32",
        f32x8::f32x8
    );
    impl_type_common!(
        f64,
        F64,
        f64::MAX,
        f64::MIN,
        0.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        "f64",
        f64x4::f64x4
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        Isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        "isize",
        isizex4::isizex4
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        Isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        "isize",
        isizex8::isizex8
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        usize,
        Usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        "usize",
        usizex4::usizex4
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        Usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        "usize",
        usizex8::usizex8
    );
    impl_type_common!(
        f16,
        F16,
        f16::MAX,
        f16::MIN,
        f16::ZERO,
        f16::ONE,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::from_f32_const(2.0),
        "f16",
        f16x16::f16x16
    );
    impl_type_common!(
        bf16,
        BF16,
        bf16::MAX,
        bf16::MIN,
        bf16::ZERO,
        bf16::ONE,
        bf16::INFINITY,
        bf16::NEG_INFINITY,
        bf16::from_f32_const(2.0),
        "bf16",
        bf16x16::bf16x16
    );
    impl_type_common!(
        Complex32,
        C32,
        Complex32::new(f32::MAX, f32::MAX),
        Complex32::new(f32::MIN, f32::MIN),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(f32::INFINITY, f32::INFINITY),
        Complex32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        Complex32::new(2.0, 0.0),
        "c32",
        cplx32x4::cplx32x4
    );
    impl_type_common!(
        Complex64,
        C64,
        Complex64::new(f64::MAX, f64::MAX),
        Complex64::new(f64::MIN, f64::MIN),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::INFINITY, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        Complex64::new(2.0, 0.0),
        "c64",
        cplx64x2::cplx64x2
    );
}

#[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
mod type_impl {
    use crate::vectors::_128bit::*;
    use half::*;
    use num_complex::{ Complex32, Complex64 };
    use super::{ TypeCommon, Dtype };

    use crate::vectors::_256bit::*;
    impl_type_common!(
        bool,
        Bool,
        true,
        false,
        false,
        true,
        true,
        false,
        false,
        "bool",
        boolx32::boolx32
    );
    impl_type_common!(i8, I8, i8::MAX, i8::MIN, 0, 1, i8::MAX, i8::MIN, 2, "i8", i8x32::i8x32);
    impl_type_common!(u8, U8, u8::MAX, u8::MIN, 0, 1, u8::MAX, u8::MIN, 2, "u8", u8x32::u8x32);
    impl_type_common!(
        i16,
        I16,
        i16::MAX,
        i16::MIN,
        0,
        1,
        i16::MAX,
        i16::MIN,
        2,
        "i16",
        i16x16::i16x16
    );
    impl_type_common!(
        u16,
        U16,
        u16::MAX,
        u16::MIN,
        0,
        1,
        u16::MAX,
        u16::MIN,
        2,
        "u16",
        u16x16::u16x16
    );
    impl_type_common!(
        i32,
        I32,
        i32::MAX,
        i32::MIN,
        0,
        1,
        i32::MAX,
        i32::MIN,
        2,
        "i32",
        i32x8::i32x8
    );
    impl_type_common!(
        u32,
        U32,
        u32::MAX,
        u32::MIN,
        0,
        1,
        u32::MAX,
        u32::MIN,
        2,
        "u32",
        u32x8::u32x8
    );
    impl_type_common!(
        i64,
        I64,
        i64::MAX,
        i64::MIN,
        0,
        1,
        i64::MAX,
        i64::MIN,
        2,
        "i64",
        i64x4::i64x4
    );
    impl_type_common!(
        u64,
        U64,
        u64::MAX,
        u64::MIN,
        0,
        1,
        u64::MAX,
        u64::MIN,
        2,
        "u64",
        u64x4::u64x4
    );
    impl_type_common!(
        f32,
        F32,
        f32::MAX,
        f32::MIN,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        "f32",
        f32x8::f32x8
    );
    impl_type_common!(
        f64,
        F64,
        f64::MAX,
        f64::MIN,
        0.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        "f64",
        f64x4::f64x4
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        Isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        "isize",
        isizex4::isizex4
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        Isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        "isize",
        isizex8::isizex8
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        usize,
        Usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        "usize",
        usizex4::usizex4
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        Usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        "usize",
        usizex8::usizex8
    );
    impl_type_common!(
        f16,
        F16,
        f16::MAX,
        f16::MIN,
        f16::ZERO,
        f16::ONE,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::from_f32_const(2.0),
        "f16",
        f16x16::f16x16
    );
    impl_type_common!(
        bf16,
        BF16,
        bf16::MAX,
        bf16::MIN,
        bf16::ZERO,
        bf16::ONE,
        bf16::INFINITY,
        bf16::NEG_INFINITY,
        bf16::from_f32_const(2.0),
        "bf16",
        bf16x16::bf16x16
    );
    impl_type_common!(
        Complex32,
        C32,
        Complex32::new(f32::MAX, f32::MAX),
        Complex32::new(f32::MIN, f32::MIN),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(f32::INFINITY, f32::INFINITY),
        Complex32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        Complex32::new(2.0, 0.0),
        "c32",
        cplx32x4::cplx32x4
    );
    impl_type_common!(
        Complex64,
        C64,
        Complex64::new(f64::MAX, f64::MAX),
        Complex64::new(f64::MIN, f64::MIN),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::INFINITY, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        Complex64::new(2.0, 0.0),
        "c64",
        cplx64x2::cplx64x2
    );
}


pub trait FloatConst {
    const HALF: Self;
    const E: Self;
    const PI: Self;
    const THREE: Self;
    const SIX: Self;
    const TWOPI: Self;
    const FOURPI: Self;
    const POINT_TWO: Self;
}

impl FloatConst for f32 {
    const HALF: Self = 0.5;
    const E: Self = std::f32::consts::E;
    const PI: Self = std::f32::consts::PI;
    const THREE: Self = 3.0;
    const SIX: Self = 6.0;
    const TWOPI: Self = std::f32::consts::PI * 2.0;
    const FOURPI: Self = std::f32::consts::PI * 4.0;
    const POINT_TWO: Self = 0.2;
}

impl FloatConst for f64 {
    const HALF: Self = 0.5;
    const E: Self = std::f64::consts::E;
    const PI: Self = std::f64::consts::PI;
    const THREE: Self = 3.0;
    const SIX: Self = 6.0;
    const TWOPI: Self = std::f64::consts::PI * 2.0;
    const FOURPI: Self = std::f64::consts::PI * 4.0;
    const POINT_TWO: Self = 0.2;
}

impl FloatConst for f16 {
    const HALF: Self = f16::from_f32_const(0.5);
    const E: Self = f16::from_f32_const(std::f32::consts::E);
    const PI: Self = f16::from_f32_const(std::f32::consts::PI);
    const THREE: Self = f16::from_f32_const(3.0);
    const SIX: Self = f16::from_f32_const(6.0);
    const TWOPI: Self = f16::from_f32_const(std::f32::consts::PI * 2.0);
    const FOURPI: Self = f16::from_f32_const(std::f32::consts::PI * 4.0);
    const POINT_TWO: Self = f16::from_f32_const(0.2);
}

impl FloatConst for bf16 {
    const HALF: Self = bf16::from_f32_const(0.5);
    const E: Self = bf16::from_f32_const(std::f32::consts::E);
    const PI: Self = bf16::from_f32_const(std::f32::consts::PI);
    const THREE: Self = bf16::from_f32_const(3.0);
    const SIX: Self = bf16::from_f32_const(6.0);
    const TWOPI: Self = bf16::from_f32_const(std::f32::consts::PI * 2.0);
    const FOURPI: Self = bf16::from_f32_const(std::f32::consts::PI * 4.0);
    const POINT_TWO: Self = bf16::from_f32_const(0.2);
}

impl NormalOut for Dtype {
    type Output = Dtype;

    fn _add(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _sub(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _mul(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _pow(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _rem(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _max(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _min(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _square(self) -> Self::Output {
        self
    }

    fn _abs(self) -> Self::Output {
        self
    }
    fn _ceil(self) -> Self::Output {
        self
    }
    fn _floor(self) -> Self::Output {
        self
    }

    fn _sign(self) -> Self::Output {
        self
    }

    fn _clip(self, min: Self::Output, _: Self::Output) -> Self::Output {
        infer_enum_type!(self, min, normal)
    }

    fn _round(self) -> Self::Output {
        self
    }

    fn _neg(self) -> Self {
        self
    }

    fn _mul_add(self, a: Self, _: Self) -> Self::Output {
        infer_enum_type!(self, a, normal)
    }
}

impl BitWiseOut for Dtype {
    type Output = Dtype;

    fn _bitand(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _bitor(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _bitxor(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _not(self) -> Self::Output {
        self
    }

    fn _shl(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _shr(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }
}

impl FloatOutUnary for Dtype {
    type Output = Dtype;
    type Base = Self;
    fn _exp(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _exp2(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _ln(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _log2(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _log10(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _sqrt(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _sin(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _cos(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _tan(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _asin(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _acos(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _atan(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _sinh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _cosh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _tanh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _asinh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _acosh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _atanh(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _recip(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _erf(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _celu(self, _: Self::Output) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _sigmoid(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _elu(self, _: Self::Output) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _leaky_relu(self, _: Self::Output) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _relu(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _gelu(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _selu(self, _: Self, _: Self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _hard_sigmoid(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _relu6(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _hard_swish(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _softplus(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
    fn _softsign(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }

    fn _mish(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }

    fn _cbrt(self) -> Self::Output {
        infer_enum_type!(self, null, uary_float)
    }
}

impl FloatOutBinary for Dtype {
    type Output = Dtype;

    fn _div(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, binary_float)
    }
    fn _log(self, base: Self) -> Self::Output {
        infer_enum_type!(self, base, binary_float)
    }
}

impl Eval for Dtype {
    type Output = Dtype;
    fn _is_nan(&self) -> Dtype {
        Dtype::Bool
    }

    fn _is_true(&self) -> Dtype {
        Dtype::Bool
    }

    fn _is_inf(&self) -> Self::Output {
        Dtype::Bool
    }
}
