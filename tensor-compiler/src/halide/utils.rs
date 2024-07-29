use tensor_types::dtype::Dtype;

use super::{ exprs::{ BitAnd, Call, Float, Int }, prime_expr::PrimeExpr };

pub fn all(conds: &[PrimeExpr]) -> PrimeExpr {
    conds
        .into_iter()
        .cloned()
        .reduce(|acc, x| BitAnd::make(acc, x).into())
        .unwrap()
}

pub fn exp(x: PrimeExpr) -> PrimeExpr {
    Call::make("exp", &[&x]).into()
}

pub fn dtype_zero(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 0)),
        Dtype::I8 => (0i8).into(),
        Dtype::U8 => (0u8).into(),
        Dtype::I16 => (0i16).into(),
        Dtype::U16 => (0u16).into(),
        Dtype::I32 => (0i32).into(),
        Dtype::U32 => (0u32).into(),
        Dtype::I64 => (0i64).into(),
        Dtype::U64 => (0u64).into(),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, 0.0)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::BF16, 0.0)),
        Dtype::F32 => (0f32).into(),
        Dtype::F64 => (0f64).into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        Dtype::Isize => (0isize).into(),
        Dtype::Usize => (0usize).into(),
    }
}

pub fn dtype_one(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 1)),
        Dtype::I8 => (1i8).into(),
        Dtype::U8 => (1u8).into(),
        Dtype::I16 => (1i16).into(),
        Dtype::U16 => (1u16).into(),
        Dtype::I32 => (1i32).into(),
        Dtype::U32 => (1u32).into(),
        Dtype::I64 => (1i64).into(),
        Dtype::U64 => (1u64).into(),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, 1.0)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::BF16, 1.0)),
        Dtype::F32 => (1f32).into(),
        Dtype::F64 => (1f64).into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        Dtype::Isize => (1isize).into(),
        Dtype::Usize => (1usize).into(),
    }
}