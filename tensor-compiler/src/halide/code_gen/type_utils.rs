use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    types::values::BasicValue,
    BasicType,
};
use tensor_types::dtype::Dtype;

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
    origin_dtype: Dtype,
    target_dtype: Dtype,
    val: BasicValue,
    var_name: &str,
    context: &Context,
    builder: &Builder
) -> BasicValue {
    let res_type = dtype_to_llvm(target_dtype, context);
    match val {
        BasicValue::F64(_) | BasicValue::F32(_) =>
            match origin_dtype {
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
            match origin_dtype {
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
            match origin_dtype {
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
            match origin_dtype {
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
        _ => panic!("{}", &format!("Unsupported cast {:?}", val)),
    }
}
