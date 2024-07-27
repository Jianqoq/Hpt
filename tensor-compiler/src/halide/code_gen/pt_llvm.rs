use tensor_llvm::{
    context::context::Context,
    types::{ general_types::GeneralType, values::StructValue },
};
use tensor_types::dtype::Dtype;

use crate::halide::primitive_type::PrimitiveType;

pub fn primitive_ty_to_llvm(
    ctx: &Context,
    ty: &PrimitiveType,
    tensor_ty: StructValue
) -> GeneralType {
    match ty {
        PrimitiveType::Dtype(dtype) => {
            {
                let dtype = *dtype;
                let context: &Context = &ctx;
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
        }
        PrimitiveType::Tuple(_) => todo!(),
        PrimitiveType::Array(_) => todo!(),
        PrimitiveType::Ptr(ptr) => {
            match ptr.inner.as_ref() {
                PrimitiveType::Dtype(dtype) => {
                    match dtype {
                        Dtype::Bool => { ctx.bool_type().ptr_type(0).into() }
                        Dtype::I8 => { ctx.i8_type().ptr_type(0).into() }
                        Dtype::U8 => { ctx.u8_type().ptr_type(0).into() }
                        Dtype::I16 => { ctx.i16_type().ptr_type(0).into() }
                        Dtype::U16 => { ctx.u16_type().ptr_type(0).into() }
                        Dtype::I32 => { ctx.i32_type().ptr_type(0).into() }
                        Dtype::U32 => { ctx.u32_type().ptr_type(0).into() }
                        Dtype::I64 => { ctx.i64_type().ptr_type(0).into() }
                        Dtype::U64 => { ctx.u64_type().ptr_type(0).into() }
                        Dtype::BF16 => { ctx.i16_type().ptr_type(0).into() }
                        Dtype::F16 => { ctx.f16_type().ptr_type(0).into() }
                        Dtype::F32 => { ctx.f32_type().ptr_type(0).into() }
                        Dtype::F64 => { ctx.f64_type().ptr_type(0).into() }
                        Dtype::C32 => todo!(),
                        Dtype::C64 => todo!(),
                        Dtype::Isize => { ctx.isize_type().ptr_type(0).into() }
                        Dtype::Usize => todo!(),
                    }
                }
                PrimitiveType::Tuple(tuple) => {
                    let types = tuple.inner
                        .iter()
                        .map(|x| x.to_llvm_type(ctx, tensor_ty))
                        .collect::<Vec<_>>();
                    ctx.struct_type(&types, false).into()
                }
                PrimitiveType::Array(_) => todo!(),
                PrimitiveType::Ptr(_) => { ctx.void_type().ptr_type(0).into() }
                PrimitiveType::Tensor(_) => todo!(),
                PrimitiveType::Str => { ctx.str_ptr_type().into() }
                PrimitiveType::Void => { ctx.void_type().ptr_type(0).into() }
            }
        }
        PrimitiveType::Tensor(_) => todo!(),
        PrimitiveType::Str => todo!(),
        PrimitiveType::Void => todo!(),
    }
}
