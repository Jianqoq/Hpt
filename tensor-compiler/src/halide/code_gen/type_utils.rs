use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    types::{ general_types::GeneralType, values::BasicValue },
    BasicType,
};
use tensor_types::dtype::Dtype;

use crate::halide::primitive_type::{ PrimitiveType, Ptr };

pub fn dtype_to_llvm(dtype: Dtype, context: &Context) -> BasicType {
    match dtype {
        Dtype::Bool => context.bool_type().into(),
        Dtype::I8 => context.i8_type().into(),
        Dtype::U8 => context.i8_type().into(),
        Dtype::I16 => context.i16_type().into(),
        Dtype::U16 => context.i16_type().into(),
        Dtype::I32 => context.i32_type().into(),
        Dtype::U32 => context.i32_type().into(),
        Dtype::I64 => context.i64_type().into(),
        Dtype::U64 => context.i64_type().into(),
        Dtype::BF16 => context.i16_type().into(),
        Dtype::F16 => context.f16_type().into(),
        Dtype::F32 => context.f32_type().into(),
        Dtype::F64 => context.f64_type().into(),
        Dtype::C32 => context.f64_type().into(),
        Dtype::C64 => context.f64_type().into(),
        Dtype::Isize => context.i64_type().into(),
        Dtype::Usize => context.i64_type().into(),
    }
}

pub fn build_cast(
    target_dtype: Dtype,
    val: BasicValue,
    var_name: &str,
    context: &Context,
    builder: &Builder
) -> BasicValue {
    let res_type = dtype_to_llvm(target_dtype, context);
    match val {
        BasicValue::F64(_) | BasicValue::F32(_) =>
            match target_dtype {
                Dtype::F64 => builder.build_float_cast(res_type, val, var_name),
                Dtype::F32 => builder.build_float_cast(res_type, val, var_name),
                Dtype::F16 => builder.build_float_cast(res_type, val, var_name),
                Dtype::Bool => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I8 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U8 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I16 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U16 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I32 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U32 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I64 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U64 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                _ => panic!("Unsupported cast"),
            }
        BasicValue::I64(_) | BasicValue::I32(_) | BasicValue::I16(_) | BasicValue::I8(_) => {
            match target_dtype {
                Dtype::F64 => builder.build_signed_int_to_float(res_type, val, var_name),
                Dtype::F32 => builder.build_signed_int_to_float(res_type, val, var_name),
                Dtype::Bool => builder.build_signed_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I8 => builder.build_signed_int_to_signed_int(res_type, val, var_name),
                Dtype::U8 => builder.build_signed_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I16 => builder.build_signed_int_to_signed_int(res_type, val, var_name),
                Dtype::U16 => builder.build_signed_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I32 => builder.build_signed_int_to_signed_int(res_type, val, var_name),
                Dtype::U32 => builder.build_signed_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I64 => builder.build_signed_int_to_signed_int(res_type, val, var_name),
                Dtype::U64 => builder.build_signed_int_to_unsigned_int(res_type, val, var_name),
                _ => panic!("Unsupported cast"),
            }
        }
        BasicValue::U8(_) | BasicValue::U16(_) | BasicValue::U32(_) | BasicValue::U64(_) => {
            match target_dtype {
                Dtype::F64 => builder.build_unsigned_int_to_float(res_type, val, var_name),
                Dtype::F32 => builder.build_unsigned_int_to_float(res_type, val, var_name),
                Dtype::Bool => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I8 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U8 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I16 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U16 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I32 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U32 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I64 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U64 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                _ => panic!("Unsupported cast"),
            }
        }
        BasicValue::F16(_) =>
            match target_dtype {
                Dtype::F64 => builder.build_float_cast(res_type, val, var_name),
                Dtype::F32 => builder.build_float_cast(res_type, val, var_name),
                Dtype::F16 => builder.build_float_cast(res_type, val, var_name),
                Dtype::Bool => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I8 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U8 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I16 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U16 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I32 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U32 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                Dtype::I64 => builder.build_float_to_signed_int(res_type, val, var_name),
                Dtype::U64 => builder.build_float_to_unsigned_int(res_type, val, var_name),
                _ => panic!("Unsupported cast"),
            }
        BasicValue::Bool(_) =>
            match target_dtype {
                Dtype::Bool => val,
                Dtype::I8 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U8 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I16 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U16 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I32 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U32 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::I64 => builder.build_unsigned_int_to_signed_int(res_type, val, var_name),
                Dtype::U64 => builder.build_unsigned_int_to_unsigned_int(res_type, val, var_name),
                Dtype::F64 => builder.build_unsigned_int_to_float(res_type, val, var_name),
                Dtype::F32 => builder.build_unsigned_int_to_float(res_type, val, var_name),
                Dtype::F16 => builder.build_unsigned_int_to_float(res_type, val, var_name),
                _ => panic!("{}", &format!("Unsupported cast {:?}", val)),
            }
        _ => panic!("{}", &format!("Unsupported cast {:?}", val)),
    }
}

pub fn general_types_to_primitive_type(general_type: &GeneralType) -> PrimitiveType {
    match general_type {
        GeneralType::Bool(_) => PrimitiveType::Dtype(Dtype::Bool),
        GeneralType::I8(_) => PrimitiveType::Dtype(Dtype::I8),
        GeneralType::U8(_) => PrimitiveType::Dtype(Dtype::U8),
        GeneralType::I16(_) => PrimitiveType::Dtype(Dtype::I16),
        GeneralType::U16(_) => PrimitiveType::Dtype(Dtype::U16),
        GeneralType::I32(_) => PrimitiveType::Dtype(Dtype::I32),
        GeneralType::U32(_) => PrimitiveType::Dtype(Dtype::U32),
        GeneralType::I64(_) => PrimitiveType::Dtype(Dtype::I64),
        GeneralType::U64(_) => PrimitiveType::Dtype(Dtype::U64),
        GeneralType::BF16(_) => PrimitiveType::Dtype(Dtype::BF16),
        GeneralType::F16(_) => PrimitiveType::Dtype(Dtype::F16),
        GeneralType::F32(_) => PrimitiveType::Dtype(Dtype::F32),
        GeneralType::F64(_) => PrimitiveType::Dtype(Dtype::F64),
        GeneralType::Void(_) => PrimitiveType::Void,
        GeneralType::Isize(_) => PrimitiveType::Dtype(Dtype::Isize),
        GeneralType::Usize(_) => PrimitiveType::Dtype(Dtype::Usize),
        GeneralType::Array(_) => todo!(),
        GeneralType::BoolPtr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::Bool).into() }),
        GeneralType::I8Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::I8).into() }),
        GeneralType::U8Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::U8).into() }),
        GeneralType::I16Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::I16).into() }),
        GeneralType::I64Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::I64).into() }),
        GeneralType::U16Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::U16).into() }),
        GeneralType::I32Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::I32).into() }),
        GeneralType::U32Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::U32).into() }),
        GeneralType::U64Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::U64).into() }),
        GeneralType::BF16Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::BF16).into() }),
        GeneralType::F16Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::F16).into() }),
        GeneralType::F32Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::F32).into() }),
        GeneralType::F64Ptr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::F64).into() }),
        GeneralType::VoidPtr(_) => PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Void.into() }),
        GeneralType::IsizePtr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::Isize).into() }),
        GeneralType::UsizePtr(_) =>
            PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Dtype(Dtype::Usize).into() }),
        GeneralType::ArrayPtr(_) => todo!(),
        GeneralType::FunctionPtr(_) => todo!(),
        GeneralType::Function(_) => todo!(),
        GeneralType::Str(_) => PrimitiveType::Str,
        GeneralType::StrPtr(_) => PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Str.into() }),
        GeneralType::Struct(_) => todo!(),
        GeneralType::StructPtr(_) => todo!(),
    }
}
