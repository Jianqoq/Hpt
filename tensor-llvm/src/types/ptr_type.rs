use std::rc::Rc;

use llvm_sys::core::LLVMPointerType;
use llvm_sys::prelude::LLVMTypeRef;
use paste::paste;

use crate::FunctionType;

use super::general_types::GeneralType;
use super::info_trait::{ TypeTrait, UnitizlizeValue };
use super::ptr_values::PtrValue;
use super::values::*;

macro_rules! register_ptr_type {
    ($name:ident) => {
        #[derive(Debug, PartialEq, Copy, Clone)]
        pub struct $name {
            pub(crate) ptr_type: LLVMTypeRef,
        }

        impl From<LLVMTypeRef> for $name {
            fn from(ptr_type: LLVMTypeRef) -> Self {
                if ptr_type.is_null() {
                    panic!("Value is null");
                }
                Self { ptr_type }
            }
        }

        impl $name {
            pub fn inner(&self) -> LLVMTypeRef {
                self.ptr_type
            }

            pub fn ptr_type(&self, address_space: u32) -> PtrType {
                let new_ptr = unsafe { LLVMPointerType(self.ptr_type, address_space) };
                let new_type = $name::from(new_ptr);
                PtrType::from(new_type)
            }

            pub fn fn_type(&self, param_types: &[GeneralType], is_var_args: bool) -> FunctionType {
                let types = Rc::new(param_types.to_vec());
                let mut param_types: Vec<LLVMTypeRef> = param_types.iter().map(|t| t.inner()).collect();
                let fn_type = unsafe { llvm_sys::core::LLVMFunctionType(self.inner(), param_types.as_mut_ptr(), param_types.len() as u32, is_var_args as i32) };
                let general_types = GeneralType::from(self.clone());
                FunctionType::new(fn_type, general_types, types, param_types.len())
            }
        }
    };
}

register_ptr_type!(BoolPtrType);
register_ptr_type!(I8PtrType);
register_ptr_type!(U8PtrType);
register_ptr_type!(I16PtrType);
register_ptr_type!(U16PtrType);
register_ptr_type!(I32PtrType);
register_ptr_type!(U32PtrType);
register_ptr_type!(I64PtrType);
register_ptr_type!(U64PtrType);
register_ptr_type!(BF16PtrType);
register_ptr_type!(F16PtrType);
register_ptr_type!(F32PtrType);
register_ptr_type!(F64PtrType);
register_ptr_type!(VoidPtrType);
register_ptr_type!(IsizePtrType);
register_ptr_type!(ArrayPtrType);
register_ptr_type!(FunctionPtrType);
register_ptr_type!(StrPtrType);
register_ptr_type!(StructPtrType);

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum PtrType {
    Bool(BoolPtrType),
    I8(I8PtrType),
    U8(U8PtrType),
    I16(I16PtrType),
    U16(U16PtrType),
    I32(I32PtrType),
    U32(U32PtrType),
    I64(I64PtrType),
    U64(U64PtrType),
    BF16(BF16PtrType),
    F16(F16PtrType),
    F32(F32PtrType),
    F64(F64PtrType),
    Void(VoidPtrType),
    Isize(IsizePtrType),
    Array(ArrayPtrType),
    Function(FunctionPtrType),
    Str(StrPtrType),
    Struct(StructPtrType),
}

impl PtrType {
    pub fn inner(&self) -> LLVMTypeRef {
        match self {
            PtrType::Bool(ptr_type) => ptr_type.ptr_type,
            PtrType::I8(ptr_type) => ptr_type.ptr_type,
            PtrType::U8(ptr_type) => ptr_type.ptr_type,
            PtrType::I16(ptr_type) => ptr_type.ptr_type,
            PtrType::U16(ptr_type) => ptr_type.ptr_type,
            PtrType::I32(ptr_type) => ptr_type.ptr_type,
            PtrType::U32(ptr_type) => ptr_type.ptr_type,
            PtrType::I64(ptr_type) => ptr_type.ptr_type,
            PtrType::U64(ptr_type) => ptr_type.ptr_type,
            PtrType::BF16(ptr_type) => ptr_type.ptr_type,
            PtrType::F16(ptr_type) => ptr_type.ptr_type,
            PtrType::F32(ptr_type) => ptr_type.ptr_type,
            PtrType::F64(ptr_type) => ptr_type.ptr_type,
            PtrType::Void(ptr_type) => ptr_type.ptr_type,
            PtrType::Isize(ptr_type) => ptr_type.ptr_type,
            PtrType::Array(ptr_type) => ptr_type.ptr_type,
            PtrType::Function(ptr_type) => ptr_type.ptr_type,
            PtrType::Str(ptr_type) => ptr_type.ptr_type,
            PtrType::Struct(ptr_type) => ptr_type.ptr_type,
        }
    }
}

macro_rules! impl_from {
    ($base_name:ident) => {
        paste! {impl From<[<$base_name PtrType>]> for PtrType {
            fn from(ptr_type: [<$base_name PtrType>]) -> Self {
                PtrType::$base_name(ptr_type)
            }
        }}
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
impl_from!(Isize);
impl_from!(Array);
impl_from!(Function);
impl_from!(Str);
impl_from!(Struct);

impl TypeTrait for PtrType {
    fn get_type(&self) -> LLVMTypeRef {
        self.inner()
    }
}

impl UnitizlizeValue for PtrType {
    fn unitialize(&self) -> BasicValue {
        match self {
            PtrType::Bool(_) => BasicValue::BoolPtr(BoolPtrValue::unitialzed()),
            PtrType::I8(_) => BasicValue::I8Ptr(I8PtrValue::unitialzed()),
            PtrType::U8(_) => BasicValue::U8Ptr(U8PtrValue::unitialzed()),
            PtrType::I16(_) => BasicValue::I16Ptr(I16PtrValue::unitialzed()),
            PtrType::U16(_) => BasicValue::U16Ptr(U16PtrValue::unitialzed()),
            PtrType::I32(_) => BasicValue::I32Ptr(I32PtrValue::unitialzed()),
            PtrType::U32(_) => BasicValue::U32Ptr(U32PtrValue::unitialzed()),
            PtrType::I64(_) => BasicValue::I64Ptr(I64PtrValue::unitialzed()),
            PtrType::U64(_) => BasicValue::U64Ptr(U64PtrValue::unitialzed()),
            PtrType::BF16(_) => BasicValue::BF16Ptr(BF16PtrValue::unitialzed()),
            PtrType::F16(_) => BasicValue::F16Ptr(F16PtrValue::unitialzed()),
            PtrType::F32(_) => BasicValue::F32Ptr(F32PtrValue::unitialzed()),
            PtrType::F64(_) => BasicValue::F64Ptr(F64PtrValue::unitialzed()),
            PtrType::Void(_) => BasicValue::VoidPtr(VoidPtrValue::unitialzed()),
            PtrType::Isize(_) => BasicValue::IsizePtr(IsizePtrValue::unitialzed()),
            PtrType::Array(_) => BasicValue::ArrayPtr(ArrayPtrValue::unitialzed()),
            PtrType::Function(_) => BasicValue::FunctionPtr(FunctionPtrValue::unitialzed()),
            PtrType::Str(_) => BasicValue::StrPtr(StrPtrValue::unitialzed()),
            PtrType::Struct(_) => BasicValue::StructPtr(StructPtrValue::unitialzed()),
        }
    }
}

macro_rules! impl_type_trait {
    ($($name:ident),*) => {
        $(
            paste! {
                impl TypeTrait for [<$name PtrType>] {
                    fn get_type(&self) -> LLVMTypeRef {
                        self.inner()
                    }
                }

                impl UnitizlizeValue for [<$name PtrType>] {
                    fn unitialize(&self) -> BasicValue {
                        BasicValue::[<$name Ptr>]([<$name PtrValue>]::unitialized())
                    }
                }

                impl [<$name PtrType>] {
                    pub fn unitialize() -> BasicValue {
                        BasicValue::[<$name Ptr>]([<$name PtrValue>]::unitialized())
                    }
                }
            }
        )*
    };
}

impl_type_trait!(
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
    Isize,
    Void,
    Array,
    Function,
    Str,
    Struct
);

impl From<BasicValue> for PtrValue {
    fn from(value: BasicValue) -> Self {
        match value {
            BasicValue::BoolPtr(val) => PtrValue::Bool(val),
            BasicValue::I8Ptr(val) => PtrValue::I8(val),
            BasicValue::U8Ptr(val) => PtrValue::U8(val),
            BasicValue::I16Ptr(val) => PtrValue::I16(val),
            BasicValue::U16Ptr(val) => PtrValue::U16(val),
            BasicValue::I32Ptr(val) => PtrValue::I32(val),
            BasicValue::U32Ptr(val) => PtrValue::U32(val),
            BasicValue::I64Ptr(val) => PtrValue::I64(val),
            BasicValue::U64Ptr(val) => PtrValue::U64(val),
            BasicValue::BF16Ptr(val) => PtrValue::BF16(val),
            BasicValue::F16Ptr(val) => PtrValue::F16(val),
            BasicValue::F32Ptr(val) => PtrValue::F32(val),
            BasicValue::F64Ptr(val) => PtrValue::F64(val),
            BasicValue::VoidPtr(val) => PtrValue::Void(val),
            BasicValue::IsizePtr(val) => PtrValue::Isize(val),
            BasicValue::ArrayPtr(val) => PtrValue::Array(val),
            BasicValue::FunctionPtr(val) => PtrValue::Function(val),
            BasicValue::StrPtr(val) => PtrValue::Str(val),
            _ => panic!("{}", &format!("{:?} is not a pointer type", value)),
        }
    }
}

impl From<PtrValue> for BasicValue {
    fn from(value: PtrValue) -> Self {
        match value {
            PtrValue::Bool(val) => BasicValue::BoolPtr(val),
            PtrValue::I8(val) => BasicValue::I8Ptr(val),
            PtrValue::U8(val) => BasicValue::U8Ptr(val),
            PtrValue::I16(val) => BasicValue::I16Ptr(val),
            PtrValue::U16(val) => BasicValue::U16Ptr(val),
            PtrValue::I32(val) => BasicValue::I32Ptr(val),
            PtrValue::U32(val) => BasicValue::U32Ptr(val),
            PtrValue::I64(val) => BasicValue::I64Ptr(val),
            PtrValue::U64(val) => BasicValue::U64Ptr(val),
            PtrValue::BF16(val) => BasicValue::BF16Ptr(val),
            PtrValue::F16(val) => BasicValue::F16Ptr(val),
            PtrValue::F32(val) => BasicValue::F32Ptr(val),
            PtrValue::F64(val) => BasicValue::F64Ptr(val),
            PtrValue::Void(val) => BasicValue::VoidPtr(val),
            PtrValue::Isize(val) => BasicValue::IsizePtr(val),
            PtrValue::Array(val) => BasicValue::ArrayPtr(val),
            PtrValue::Function(val) => BasicValue::FunctionPtr(val),
            PtrValue::Str(val) => BasicValue::StrPtr(val),
            PtrValue::Struct(val) => BasicValue::StructPtr(val),
        }
    }
}
