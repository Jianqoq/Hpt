use tensor_types::dtype::Dtype;

use super::{
    exprs::{ BitAnd, Call, Float, Int, Neg },
    prime_expr::PrimeExpr,
    stmt::Stmt,
    store_stmt::StoreStmt,
    variable::Variable,
};

pub fn all(conds: &[PrimeExpr]) -> PrimeExpr {
    conds
        .into_iter()
        .cloned()
        .reduce(|acc, x| BitAnd::make(acc, x).into())
        .unwrap()
}

pub fn bitand<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(a: A, b: B) -> PrimeExpr {
    BitAnd::make(a.into(), b.into()).into()
}

pub fn exp(x: PrimeExpr) -> PrimeExpr {
    Call::make("exp", &[&x]).into()
}

pub fn erf(x: PrimeExpr) -> PrimeExpr {
    Call::make("erf", &[&x]).into()
}

pub fn neg(x: PrimeExpr) -> PrimeExpr {
    Neg::make(x).into()
}

pub fn floor(x: PrimeExpr) -> PrimeExpr {
    Call::make("floor", &[&x]).into()
}

pub fn store_with_idx<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(
    var: String,
    index: A,
    val: B
) -> Stmt {
    (StoreStmt {
        var: Variable::new(var),
        begins: vec![(0i64).into()].into(),
        axes: vec![index.into()].into(),
        steps: vec![(1i64).into()].into(),
        strides: vec![(1i64).into()].into(),
        val: val.into().into(),
    }).into()
}

pub fn store_with_dims<A: Into<PrimeExpr>, B: Into<PrimeExpr>, C: Into<PrimeExpr>>(
    var: String,
    dims: Vec<A>,
    strides: Vec<B>,
    val: C
) -> Stmt {
    (StoreStmt {
        var: Variable::new(var),
        begins: vec![(0i64).into()].into(),
        axes: dims
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<_>>()
            .into(),
        steps: vec![(1i64).into()].into(),
        strides: strides
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<_>>()
            .into(),
        val: val.into().into(),
    }).into()
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
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, 0.0)),
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
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, 1.0)),
        Dtype::F32 => (1f32).into(),
        Dtype::F64 => (1f64).into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        Dtype::Isize => (1isize).into(),
        Dtype::Usize => (1usize).into(),
    }
}

pub fn dtype_point5(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 0)),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, 0.5)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, 0.5)),
        Dtype::F32 => (0.5f32).into(),
        Dtype::F64 => (0.5f64).into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        _ => unreachable!(),
    }
}

pub fn dtype_sqrt2(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 0)),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, std::f64::consts::SQRT_2)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, std::f64::consts::SQRT_2)),
        Dtype::F32 => std::f32::consts::SQRT_2.into(),
        Dtype::F64 => std::f64::consts::SQRT_2.into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        _ => unreachable!(),
    }
}

pub fn dtype_inf(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 1)),
        Dtype::I8 => i8::MAX.into(),
        Dtype::U8 => u8::MAX.into(),
        Dtype::I16 => i16::MAX.into(),
        Dtype::U16 => u16::MAX.into(),
        Dtype::I32 => i32::MAX.into(),
        Dtype::U32 => u32::MAX.into(),
        Dtype::I64 => i64::MAX.into(),
        Dtype::U64 => u64::MAX.into(),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, f32::INFINITY as f64)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, f32::INFINITY as f64)),
        Dtype::F32 => f32::INFINITY.into(),
        Dtype::F64 => f64::INFINITY.into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        _ => unreachable!(),
    }
}

pub fn dtype_neginf(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 1)),
        Dtype::I8 => i8::MIN.into(),
        Dtype::U8 => (0u8).into(),
        Dtype::I16 => i16::MIN.into(),
        Dtype::U16 => (0u16).into(),
        Dtype::I32 => i32::MIN.into(),
        Dtype::U32 => (0u32).into(),
        Dtype::I64 => i64::MIN.into(),
        Dtype::U64 => (0u64).into(),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, f32::NEG_INFINITY as f64)),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, f32::NEG_INFINITY as f64)),
        Dtype::F32 => f32::NEG_INFINITY.into(),
        Dtype::F64 => f64::NEG_INFINITY.into(),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        _ => unreachable!(),
    }
}
