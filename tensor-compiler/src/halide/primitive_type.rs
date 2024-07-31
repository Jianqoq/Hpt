use std::{ fmt::Display, sync::Arc };
use colored::Colorize;
use tensor_llvm::{
    context::context::Context,
    types::{ general_types::GeneralType, values::StructValue },
    StructType,
};
use tensor_types::{ dtype::Dtype, type_promote::{ FloatOut, NormalOut } };

#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub enum PrimitiveType {
    Dtype(Dtype),
    Tuple(Tuple),
    Array(Array),
    Ptr(Ptr),
    Tensor(Tensor),
    Str,
    Void,
}

impl PrimitiveType {
    pub fn to_tensor(&self) -> Tensor {
        match self {
            PrimitiveType::Tensor(tensor) => tensor.clone(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Tuple {
    pub inner: Arc<Vec<PrimitiveType>>,
}

impl Tuple {
    pub fn to_llvm_type(&self, ctx: &Context, tensor_type: StructValue) -> StructType {
        let mut inner = Vec::new();
        for t in self.inner.iter() {
            inner.push(t.to_llvm_type(ctx, tensor_type));
        }
        ctx.struct_type(&inner, false)
    }
}

#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Tensor {
    pub ptr: Ptr,
    pub dtype: Dtype,
    pub shape: Array,
    pub strides: Array,
}

#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Array {
    pub inner: Arc<PrimitiveType>,
    pub size: i64,
}
#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Ptr {
    pub inner: Arc<PrimitiveType>,
}

impl PrimitiveType {
    pub fn to_llvm_type(&self, ctx: &Context, tensor_type: StructValue) -> GeneralType {
        match self {
            PrimitiveType::Dtype(dtype) => {
                match dtype {
                    Dtype::Bool => GeneralType::Bool(ctx.bool_type()),
                    Dtype::I8 => GeneralType::I8(ctx.i8_type()),
                    Dtype::U8 => GeneralType::U8(ctx.u8_type()),
                    Dtype::I16 => GeneralType::I16(ctx.i16_type()),
                    Dtype::U16 => GeneralType::U16(ctx.u16_type()),
                    Dtype::I32 => GeneralType::I32(ctx.i32_type()),
                    Dtype::U32 => GeneralType::U32(ctx.u32_type()),
                    Dtype::I64 => GeneralType::I64(ctx.i64_type()),
                    Dtype::U64 => GeneralType::U64(ctx.u64_type()),
                    Dtype::F16 => GeneralType::F16(ctx.f16_type()),
                    Dtype::F32 => GeneralType::F32(ctx.f32_type()),
                    Dtype::F64 => GeneralType::F64(ctx.f64_type()),
                    Dtype::Isize => GeneralType::Isize(ctx.isize_type()),
                    _ => unimplemented!(),
                }
            }
            PrimitiveType::Tuple(tuple) => {
                GeneralType::Struct(tuple.to_llvm_type(ctx, tensor_type))
            }
            PrimitiveType::Ptr(ptr) =>
                match ptr.inner.as_ref() {
                    PrimitiveType::Dtype(dtype) => {
                        match dtype {
                            Dtype::Bool => GeneralType::BoolPtr(ctx.bool_type().ptr_type(0)),
                            Dtype::I8 => GeneralType::I8Ptr(ctx.i8_type().ptr_type(0)),
                            Dtype::U8 => GeneralType::U8Ptr(ctx.u8_type().ptr_type(0)),
                            Dtype::I16 => GeneralType::I16Ptr(ctx.i16_type().ptr_type(0)),
                            Dtype::U16 => GeneralType::U16Ptr(ctx.u16_type().ptr_type(0)),
                            Dtype::I32 => GeneralType::I32Ptr(ctx.i32_type().ptr_type(0)),
                            Dtype::U32 => GeneralType::U32Ptr(ctx.u32_type().ptr_type(0)),
                            Dtype::I64 => GeneralType::I64Ptr(ctx.i64_type().ptr_type(0)),
                            Dtype::U64 => GeneralType::U64Ptr(ctx.u64_type().ptr_type(0)),
                            Dtype::F16 => GeneralType::F16Ptr(ctx.f16_type().ptr_type(0)),
                            Dtype::F32 => GeneralType::F32Ptr(ctx.f32_type().ptr_type(0)),
                            Dtype::F64 => GeneralType::F64Ptr(ctx.f64_type().ptr_type(0)),
                            Dtype::Isize => GeneralType::IsizePtr(ctx.isize_type().ptr_type(0)),
                            _ => unimplemented!("unimplemented to_llvm_type for {}", dtype),
                        }
                    }
                    PrimitiveType::Tuple(tuple) => {
                        let mut types = vec![];
                        for i in tuple.inner.iter() {
                            types.push(i.to_llvm_type(ctx, tensor_type));
                        }
                        GeneralType::StructPtr(ctx.struct_type(&types, false).ptr_type(0))
                    }
                    PrimitiveType::Array(_) => todo!(),
                    PrimitiveType::Ptr(ptr) => {
                        match ptr.inner.as_ref() {
                            PrimitiveType::Dtype(dtype) => {
                                match dtype {
                                    Dtype::Bool => {
                                        GeneralType::BoolPtr(ctx.bool_type().ptr_type(0))
                                    }
                                    Dtype::I8 => { GeneralType::I8Ptr(ctx.i8_type().ptr_type(0)) }
                                    Dtype::U8 => { GeneralType::U8Ptr(ctx.u8_type().ptr_type(0)) }
                                    Dtype::I16 => {
                                        GeneralType::I16Ptr(ctx.i16_type().ptr_type(0))
                                    }
                                    Dtype::U16 => {
                                        GeneralType::U16Ptr(ctx.u16_type().ptr_type(0))
                                    }
                                    Dtype::I32 => {
                                        GeneralType::I32Ptr(ctx.i32_type().ptr_type(0))
                                    }
                                    Dtype::U32 => {
                                        GeneralType::U32Ptr(ctx.u32_type().ptr_type(0))
                                    }
                                    Dtype::I64 => {
                                        GeneralType::I64Ptr(ctx.i64_type().ptr_type(0))
                                    }
                                    Dtype::U64 => {
                                        GeneralType::U64Ptr(ctx.u64_type().ptr_type(0))
                                    }
                                    Dtype::BF16 => todo!(),
                                    Dtype::F16 => todo!(),
                                    Dtype::F32 => {
                                        GeneralType::F32Ptr(ctx.f32_type().ptr_type(0))
                                    }
                                    Dtype::F64 => {
                                        GeneralType::F64Ptr(ctx.f64_type().ptr_type(0))
                                    }
                                    Dtype::C32 => todo!(),
                                    Dtype::C64 => todo!(),
                                    Dtype::Isize => {
                                        GeneralType::IsizePtr(ctx.isize_type().ptr_type(0))
                                    }
                                    Dtype::Usize => {
                                        GeneralType::UsizePtr(ctx.usize_type().ptr_type(0))
                                    }
                                }
                            }
                            PrimitiveType::Tuple(_) => todo!(),
                            PrimitiveType::Array(_) => todo!(),
                            PrimitiveType::Ptr(_) => todo!(),
                            PrimitiveType::Tensor(_) => todo!(),
                            PrimitiveType::Str => todo!(),
                            PrimitiveType::Void => {
                                GeneralType::VoidPtr(ctx.void_type().ptr_type(0))
                            }
                        }
                    }
                    PrimitiveType::Str => todo!(),
                    PrimitiveType::Void => todo!(),
                    PrimitiveType::Tensor(_) => {
                        GeneralType::StructPtr(ctx.tensor_type().ptr_type(0))
                    }
                }
            PrimitiveType::Tensor(_) => { GeneralType::Struct(tensor_type.to_type()) }
            PrimitiveType::Void => { GeneralType::Void(ctx.void_type()) }
            _ => unimplemented!("unimplemented to_llvm_type for {}", self),
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            PrimitiveType::Dtype(dtype) => *dtype,
            _ => unimplemented!("unimplemented dtype for {}", self),
        }
    }
    pub fn _add(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._add(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _mul(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._mul(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _sub(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._sub(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _div(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._div(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _sin(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._sin()),
            _ => unimplemented!(),
        }
    }
    pub fn _cos(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._cos()),
            _ => unimplemented!(),
        }
    }
    pub fn _tan(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._tan()),
            _ => unimplemented!(),
        }
    }
    pub fn _asin(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._asin()),
            _ => unimplemented!(),
        }
    }
    pub fn _acos(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._acos()),
            _ => unimplemented!(),
        }
    }
    pub fn _atan(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._atan()),
            _ => unimplemented!(),
        }
    }
    pub fn _exp(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._exp()),
            _ => unimplemented!(),
        }
    }
    pub fn _log(&self, base: &Self) -> Self {
        match (self, base) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._log(*rhs))
            }
            _ => unimplemented!(),
        }
    }
}

impl Display for PrimitiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ret = match self {
            PrimitiveType::Dtype(dtype) => format!("{}", dtype),
            PrimitiveType::Tuple(tuple) => {
                let mut string = String::new();
                if tuple.inner.is_empty() {
                    string.push_str(&format!("()"));
                }
                if tuple.inner.len() == 1 {
                    string.push_str(&format!("({}, )", tuple.inner[0]))
                }
                string.push_str(&format!("({}", tuple.inner[0]));
                for i in 1..tuple.inner.len() {
                    string.push_str(&format!(", {}", tuple.inner[i]))
                }
                string.push_str(&format!(")"));
                string
            }
            PrimitiveType::Array(arr) => format!("[{}; {}]", arr.inner, arr.size),
            PrimitiveType::Ptr(ptr) => format!("*{}", ptr.inner),
            PrimitiveType::Str => format!("str"),
            PrimitiveType::Void => format!("void"),
            PrimitiveType::Tensor(tensor) => { format!("Tensor<{}>", tensor.dtype) }
        };
        write!(f, "{}", ret.truecolor(230, 200, 90))
    }
}
