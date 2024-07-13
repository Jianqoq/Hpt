use llvm_sys::prelude::LLVMTypeRef;
use paste::paste;
use crate::*;
use super::ptr_type::*;

#[derive(Debug, PartialEq, Clone)]
pub enum GeneralType {
    Bool(BoolType),
    BoolPtr(BoolPtrType),
    I8(I8Type),
    I8Ptr(I8PtrType),
    U8(U8Type),
    U8Ptr(U8PtrType),
    I16(I16Type),
    I16Ptr(I16PtrType),
    U16(U16Type),
    U16Ptr(U16PtrType),
    I32(I32Type),
    I32Ptr(I32PtrType),
    U32(U32Type),
    U32Ptr(U32PtrType),
    I64(I64Type),
    I64Ptr(I64PtrType),
    U64(U64Type),
    U64Ptr(U64PtrType),
    BF16(BF16Type),
    BF16Ptr(BF16PtrType),
    F16(F16Type),
    F16Ptr(F16PtrType),
    F32(F32Type),
    F32Ptr(F32PtrType),
    F64(F64Type),
    F64Ptr(F64PtrType),
    Void(VoidType),
    VoidPtr(VoidPtrType),
    Isize(IsizeType),
    IsizePtr(IsizePtrType),
    Function(FunctionType),
    FunctionPtr(FunctionPtrType),
    Array(ArrayType),
    ArrayPtr(ArrayPtrType),
    Str(StrType),
    StrPtr(StrPtrType),
    Struct(StructType),
    StructPtr(StructPtrType),
}

macro_rules! impl_from {
    ($base_name:ident) => {
        paste! {
            impl From<[<$base_name Type>]> for GeneralType {
                fn from(t: [<$base_name Type>]) -> Self {
                    GeneralType::$base_name(t)
                }
            }
        }
    };
}

impl_from!(Bool);
impl_from!(BoolPtr);
impl_from!(I8);
impl_from!(I8Ptr);
impl_from!(U8);
impl_from!(U8Ptr);
impl_from!(I16);
impl_from!(I16Ptr);
impl_from!(U16);
impl_from!(U16Ptr);
impl_from!(I32);
impl_from!(I32Ptr);
impl_from!(U32);
impl_from!(U32Ptr);
impl_from!(I64);
impl_from!(I64Ptr);
impl_from!(U64);
impl_from!(U64Ptr);
impl_from!(BF16);
impl_from!(BF16Ptr);
impl_from!(F16);
impl_from!(F16Ptr);
impl_from!(F32);
impl_from!(F32Ptr);
impl_from!(F64);
impl_from!(F64Ptr);
impl_from!(Void);
impl_from!(VoidPtr);
impl_from!(Isize);
impl_from!(IsizePtr);
impl_from!(Function);
impl_from!(FunctionPtr);
impl_from!(Array);
impl_from!(ArrayPtr);
impl_from!(Str);
impl_from!(StrPtr);
impl_from!(Struct);
impl_from!(StructPtr);

impl GeneralType {
    pub fn inner(&self) -> LLVMTypeRef {
        match self {
            GeneralType::Bool(t) => t.inner(),
            GeneralType::BoolPtr(t) => t.inner(),
            GeneralType::I8(t) => t.inner(),
            GeneralType::I8Ptr(t) => t.inner(),
            GeneralType::U8(t) => t.inner(),
            GeneralType::U8Ptr(t) => t.inner(),
            GeneralType::I16(t) => t.inner(),
            GeneralType::I16Ptr(t) => t.inner(),
            GeneralType::U16(t) => t.inner(),
            GeneralType::U16Ptr(t) => t.inner(),
            GeneralType::I32(t) => t.inner(),
            GeneralType::I32Ptr(t) => t.inner(),
            GeneralType::U32(t) => t.inner(),
            GeneralType::U32Ptr(t) => t.inner(),
            GeneralType::I64(t) => t.inner(),
            GeneralType::I64Ptr(t) => t.inner(),
            GeneralType::U64(t) => t.inner(),
            GeneralType::U64Ptr(t) => t.inner(),
            GeneralType::BF16(t) => t.inner(),
            GeneralType::BF16Ptr(t) => t.inner(),
            GeneralType::F16(t) => t.inner(),
            GeneralType::F16Ptr(t) => t.inner(),
            GeneralType::F32(t) => t.inner(),
            GeneralType::F32Ptr(t) => t.inner(),
            GeneralType::F64(t) => t.inner(),
            GeneralType::F64Ptr(t) => t.inner(),
            GeneralType::Void(t) => t.inner(),
            GeneralType::VoidPtr(t) => t.inner(),
            GeneralType::Isize(t) => t.inner(),
            GeneralType::IsizePtr(t) => t.inner(),
            GeneralType::Function(t) => t.inner(),
            GeneralType::FunctionPtr(t) => t.inner(),
            GeneralType::Array(t) => t.inner(),
            GeneralType::ArrayPtr(t) => t.inner(),
            GeneralType::Str(t) => t.inner(),
            GeneralType::StrPtr(t) => t.inner(),
            GeneralType::Struct(t) => t.inner(),
            GeneralType::StructPtr(t) => t.inner(),
        }
    }
}
