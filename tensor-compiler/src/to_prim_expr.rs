use tensor_types::dtype::Dtype;

use crate::halide::{ exprs::{ Float, Int, UInt }, prime_expr::PrimeExpr, variable::Variable };

pub trait ToPrimeExpr {
    fn to_prime_expr(&self) -> PrimeExpr;
}

impl ToPrimeExpr for i8 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I8, *self as i64))
    }
}

impl ToPrimeExpr for &i8 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I8, **self as i64))
    }
}

impl ToPrimeExpr for i16 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I16, *self as i64))
    }
}

impl ToPrimeExpr for &i16 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I16, **self as i64))
    }
}

impl ToPrimeExpr for i32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I32, *self as i64))
    }
}

impl ToPrimeExpr for &i32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I32, **self as i64))
    }
}

impl ToPrimeExpr for i64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I64, *self))
    }
}

impl ToPrimeExpr for &i64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I64, **self))
    }
}

impl ToPrimeExpr for u8 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U8, *self as u64))
    }
}

impl ToPrimeExpr for &u8 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U8, **self as u64))
    }
}

impl ToPrimeExpr for u16 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U16, *self as u64))
    }
}

impl ToPrimeExpr for &u16 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U16, **self as u64))
    }
}

impl ToPrimeExpr for u32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U32, *self as u64))
    }
}

impl ToPrimeExpr for &u32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U32, **self as u64))
    }
}

impl ToPrimeExpr for u64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U64, *self))
    }
}

impl ToPrimeExpr for &u64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U64, **self))
    }
}

impl ToPrimeExpr for f32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F32, *self as f64))
    }
}

impl ToPrimeExpr for &f32 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F32, **self as f64))
    }
}

impl ToPrimeExpr for f64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F64, *self))
    }
}

impl ToPrimeExpr for &f64 {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F64, **self))
    }
}

impl ToPrimeExpr for Variable {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Variable(self.clone())
    }
}

impl ToPrimeExpr for &Variable {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Variable((*self).clone())
    }
}

impl ToPrimeExpr for Int {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int(self.clone())
    }
}

impl ToPrimeExpr for &Int {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Int((*self).clone())
    }
}

impl ToPrimeExpr for UInt {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt(self.clone())
    }
}

impl ToPrimeExpr for &UInt {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::UInt((*self).clone())
    }
}

impl ToPrimeExpr for Float {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float(self.clone())
    }
}

impl ToPrimeExpr for &Float {
    fn to_prime_expr(&self) -> PrimeExpr {
        PrimeExpr::Float((*self).clone())
    }
}

impl ToPrimeExpr for PrimeExpr {
    fn to_prime_expr(&self) -> PrimeExpr {
        self.clone()
    }
}

impl ToPrimeExpr for &PrimeExpr {
    fn to_prime_expr(&self) -> PrimeExpr {
        (*self).clone()
    }
}
