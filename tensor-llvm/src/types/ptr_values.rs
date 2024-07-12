use llvm_sys::prelude::LLVMValueRef;

use crate::types::values::*;
use paste::paste;
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum PtrValue {
    Bool(BoolPtrValue),
    I8(I8PtrValue),
    U8(U8PtrValue),
    I16(I16PtrValue),
    U16(U16PtrValue),
    I32(I32PtrValue),
    U32(U32PtrValue),
    I64(I64PtrValue),
    U64(U64PtrValue),
    BF16(BF16PtrValue),
    F16(F16PtrValue),
    F32(F32PtrValue),
    F64(F64PtrValue),
    Void(VoidPtrValue),
    Isize(IsizePtrValue),
    Array(ArrayPtrValue),
    Function(FunctionPtrValue),
    Str(StrPtrValue),
}

macro_rules! impl_from {
    ($base_name:ident) => {
        paste! {impl From<[<$base_name PtrValue>]> for PtrValue {
                fn from(value: [<$base_name PtrValue>]) -> Self {
                    PtrValue::$base_name(value)
                }
            }
        }
    };
}

impl_from!(Bool);
impl_from!(I8);
impl_from!(U8);
impl_from!(I16);
impl_from!(U16);
impl_from!(I32);
impl_from!(U32);
impl_from!(I64);
impl_from!(U64);
impl_from!(BF16);
impl_from!(F16);
impl_from!(F32);
impl_from!(F64);
impl_from!(Void);

impl PtrValue {
    pub fn inner(&self) -> LLVMValueRef {
        match self {
            PtrValue::Bool(ptr_value) => ptr_value.value,
            PtrValue::I8(ptr_value) => ptr_value.value,
            PtrValue::U8(ptr_value) => ptr_value.value,
            PtrValue::I16(ptr_value) => ptr_value.value,
            PtrValue::U16(ptr_value) => ptr_value.value,
            PtrValue::I32(ptr_value) => ptr_value.value,
            PtrValue::U32(ptr_value) => ptr_value.value,
            PtrValue::I64(ptr_value) => ptr_value.value,
            PtrValue::U64(ptr_value) => ptr_value.value,
            PtrValue::BF16(ptr_value) => ptr_value.value,
            PtrValue::F16(ptr_value) => ptr_value.value,
            PtrValue::F32(ptr_value) => ptr_value.value,
            PtrValue::F64(ptr_value) => ptr_value.value,
            PtrValue::Void(ptr_value) => ptr_value.value,
            PtrValue::Isize(ptr_value) => ptr_value.value,
            PtrValue::Array(ptr_value) => ptr_value.value,
            PtrValue::Function(ptr_value) => ptr_value.value,
            PtrValue::Str(ptr_value) => ptr_value.value,
        }
    }

    pub unsafe fn set_inner(&mut self, value: LLVMValueRef) {
        match self {
            PtrValue::Bool(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::I8(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::U8(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::I16(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::U16(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::I32(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::U32(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::I64(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::U64(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::BF16(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::F16(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::F32(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::F64(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::Void(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::Isize(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::Array(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::Function(ptr_value) => {
                ptr_value.value = value;
            }
            PtrValue::Str(ptr_value) => {
                ptr_value.value = value;
            }
        }
    }
}
