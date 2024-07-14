use std::fmt::Display;
use std::rc::Rc;

use crate::types::general_types::GeneralType;
use crate::types::ptr_type::*;
use crate::types::values::*;
use llvm_sys::prelude::LLVMTypeRef;
use llvm_sys::prelude::LLVMValueRef;
use paste::paste;

use super::info_trait::TypeTrait;
use super::info_trait::UnitizlizeValue;

macro_rules! register_types {
    ($($name:ident),*) => {
        $(
            paste! {

            #[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
            pub struct [<$name Type>] {
                pub(crate) value: LLVMTypeRef,
            }

            impl From<LLVMTypeRef> for [<$name Type>] {
                fn from(value: LLVMTypeRef) -> Self {
                    if value.is_null() {
                        panic!("Value is null");
                    }
                    Self { value }
                }
            }

            impl [<$name Type>] {
                pub fn inner(&self) -> LLVMTypeRef {
                    self.value
                }

                pub fn array_type(&self, size: u64) -> ArrayType {
                    let array_type = unsafe { llvm_sys::core::LLVMArrayType2(self.value, size) };
                    ArrayType::from(array_type)
                }

                pub fn ptr_type(&self, address_space: u32) -> [<$name PtrType>] {
                    let ptr_type = unsafe { llvm_sys::core::LLVMPointerType(self.value, address_space) };
                    [<$name PtrType>]::from(ptr_type)
                }

                pub fn fn_type(&self, param_types: &[GeneralType], is_var_args: bool) -> FunctionType {
                    let types = Rc::new(param_types.to_vec());
                    let mut param_types: Vec<LLVMTypeRef> = param_types.iter().map(|t| t.inner()).collect();
                    let fn_type = unsafe { llvm_sys::core::LLVMFunctionType(self.value, param_types.as_mut_ptr(), param_types.len() as u32, is_var_args as i32) };
                    let general_types = GeneralType::from(self.clone());
                    FunctionType::new(fn_type, general_types, types, param_types.len())
                }
            }
        }
        )*
    };
}

macro_rules! impl_into_basic_type {
    ($($name:ident),*) => {
        $(
            paste! {
                impl From<[<$name Type>]> for BasicType {
                    fn from(t: [<$name Type>]) -> Self {
                        BasicType::$name(t)
                    }
                }
            }
        )*
    };
}

macro_rules! impl_const_int_value {
    ($($name:ident),*) => {
        $(
            paste! {
                impl [<$name Type>] {
                    pub fn const_int(&self, value: u64, sign_extend: bool) -> [<$name Value>] {
                        let value = unsafe { llvm_sys::core::LLVMConstInt(self.value, value, sign_extend as i32) };
                        [<$name Value>]::from(value)
                    }
                    pub fn unitialized() -> Self {
                        Self {
                            value: std::ptr::null_mut(),
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! impl_const_uint_value {
    ($($name:ident),*) => {
        $(
            paste! {
                impl [<$name Type>] {
                    pub fn const_int(&self, value: u64, sign_extend: bool) -> [<$name Value>] {
                        let value = unsafe { llvm_sys::core::LLVMConstInt(self.value, value, sign_extend as i32) };
                        [<$name Value>]::from(value)
                    }
                    pub fn unitialized() -> Self {
                        Self {
                            value: std::ptr::null_mut(),
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! impl_const_float_value {
    ($($name:ident),*) => {
        $(
            paste! {
                impl [<$name Type>] {
                    pub fn const_float(&self, value: f64) -> [<$name Value>] {
                        let value = unsafe { llvm_sys::core::LLVMConstReal(self.value, value) };
                        [<$name Value>]::from(value)
                    }
                    pub fn unitialized() -> Self {
                        Self {
                            value: std::ptr::null_mut(),
                        }
                    }
                }
            }
        )*
    };
}

register_types!(
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
    Usize,
    Void,
    Array,
    Str,
    Struct
);

impl_into_basic_type!(
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
    Usize,
    Void,
    Array
);

impl_const_int_value!(I8, I16, I32, I64, Isize);
impl_const_float_value!(F16, F32, F64);
impl_const_uint_value!(Bool, U8, U16, U32, U64);

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub struct FunctionType {
    pub(crate) value: LLVMTypeRef,
    pub(crate) ret_type: Rc<GeneralType>,
    pub(crate) param_types: Rc<Vec<GeneralType>>,
    pub(crate) param_count: usize,
}

impl FunctionType {
    pub fn new(
        value: LLVMTypeRef,
        ret_type: GeneralType,
        param_types: Rc<Vec<GeneralType>>,
        param_count: usize
    ) -> Self {
        let ret_type = Rc::new(ret_type);
        Self {
            value,
            ret_type,
            param_types,
            param_count,
        }
    }

    pub fn ret_type(&self) -> Rc<GeneralType> {
        self.ret_type.clone()
    }

    pub fn param_types(&self) -> Rc<Vec<GeneralType>> {
        self.param_types.clone()
    }

    pub fn param_count(&self) -> usize {
        self.param_count
    }

    pub fn inner(&self) -> LLVMTypeRef {
        self.value
    }
    pub fn array_type(&self, size: u64) -> ArrayType {
        let array_type = unsafe { llvm_sys::core::LLVMArrayType2(self.value, size) };
        ArrayType::from(array_type)
    }
    pub fn ptr_type(&self, address_space: u32) -> FunctionPtrType {
        let ptr_type = unsafe { llvm_sys::core::LLVMPointerType(self.value, address_space) };
        FunctionPtrType::from(ptr_type)
    }
}

impl StructType {
    pub fn const_named_struct(&self, values: &[BasicValue]) -> StructValue {
        let mut values: Vec<LLVMValueRef> = values
            .iter()
            .map(|v| v.inner())
            .collect();
        let value = unsafe {
            llvm_sys::core::LLVMConstNamedStruct(
                self.value,
                values.as_mut_ptr(),
                values.len() as u32
            )
        };
        StructValue::from(value)
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub enum BasicType {
    Bool(BoolType),
    I8(I8Type),
    U8(U8Type),
    I16(I16Type),
    U16(U16Type),
    I32(I32Type),
    U32(U32Type),
    I64(I64Type),
    U64(U64Type),
    BF16(BF16Type),
    F16(F16Type),
    F32(F32Type),
    F64(F64Type),
    Void(VoidType),
    Array(ArrayType),
    Isize(IsizeType),
    Usize(UsizeType),
    Struct(StructType),
}

impl Display for GeneralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneralType::Bool(_) => f.write_str("bool"),
            GeneralType::I8(_) => f.write_str("i8"),
            GeneralType::U8(_) => f.write_str("u8"),
            GeneralType::I16(_) => f.write_str("i16"),
            GeneralType::U16(_) => f.write_str("u16"),
            GeneralType::I32(_) => f.write_str("i32"),
            GeneralType::U32(_) => f.write_str("u32"),
            GeneralType::I64(_) => f.write_str("i64"),
            GeneralType::U64(_) => f.write_str("u64"),
            GeneralType::BF16(_) => f.write_str("bf16"),
            GeneralType::F16(_) => f.write_str("f16"),
            GeneralType::F32(_) => f.write_str("f32"),
            GeneralType::F64(_) => f.write_str("f64"),
            GeneralType::Void(_) => f.write_str("void"),
            GeneralType::Array(_) => f.write_str("array"),
            GeneralType::Struct(_) => f.write_str("struct"),
            GeneralType::StructPtr(_) => f.write_str("struct ptr"),
            GeneralType::Str(_) => f.write_str("str"),
            GeneralType::Isize(_) => f.write_str("isize"),
            GeneralType::Usize(_) => f.write_str("usize"),
            _ => write!(f, "unknown"),
        }
    }
}

impl BasicType {
    pub fn inner(&self) -> LLVMTypeRef {
        match self {
            BasicType::Bool(t) => t.inner(),
            BasicType::I8(t) => t.inner(),
            BasicType::U8(t) => t.inner(),
            BasicType::I16(t) => t.inner(),
            BasicType::U16(t) => t.inner(),
            BasicType::I32(t) => t.inner(),
            BasicType::U32(t) => t.inner(),
            BasicType::I64(t) => t.inner(),
            BasicType::U64(t) => t.inner(),
            BasicType::BF16(t) => t.inner(),
            BasicType::F16(t) => t.inner(),
            BasicType::F32(t) => t.inner(),
            BasicType::F64(t) => t.inner(),
            BasicType::Void(t) => t.inner(),
            BasicType::Array(t) => t.inner(),
            BasicType::Isize(t) => t.inner(),
            BasicType::Usize(t) => t.inner(),
            BasicType::Struct(t) => t.inner(),
        }
    }
    pub fn is_struct(&self) -> bool {
        match self {
            BasicType::Struct(_) => true,
            _ => false,
        }
    }
}

impl TypeTrait for BasicType {
    fn get_type(&self) -> LLVMTypeRef {
        self.inner()
    }
}

macro_rules! impl_type_trait {
    ($($name:ident),*) => {
        $(
            paste! {
                impl TypeTrait for [<$name Type>] {
                    fn get_type(&self) -> LLVMTypeRef {
                        self.inner()
                    }
                }

                impl UnitizlizeValue for [<$name Type>] {
                    fn unitialize(&self) -> BasicValue {
                        BasicValue::$name([<$name Value>]::unitialized())
                    }
                }

                impl [<$name Type>] {
                    pub fn unitialize() -> BasicValue {
                        BasicValue::$name([<$name Value>]::unitialized())
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
    Usize,
    Void,
    Array,
    Str,
    Struct
);

impl UnitizlizeValue for BasicType {
    fn unitialize(&self) -> BasicValue {
        match self {
            BasicType::Bool(_) => BasicValue::Bool(BoolValue::unitialized()),
            BasicType::I8(_) => BasicValue::I8(I8Value::unitialized()),
            BasicType::U8(_) => BasicValue::U8(U8Value::unitialized()),
            BasicType::I16(_) => BasicValue::I16(I16Value::unitialized()),
            BasicType::U16(_) => BasicValue::U16(U16Value::unitialized()),
            BasicType::I32(_) => BasicValue::I32(I32Value::unitialized()),
            BasicType::U32(_) => BasicValue::U32(U32Value::unitialized()),
            BasicType::I64(_) => BasicValue::I64(I64Value::unitialized()),
            BasicType::U64(_) => BasicValue::U64(U64Value::unitialized()),
            BasicType::BF16(_) => BasicValue::BF16(BF16Value::unitialized()),
            BasicType::F16(_) => BasicValue::F16(F16Value::unitialized()),
            BasicType::F32(_) => BasicValue::F32(F32Value::unitialized()),
            BasicType::F64(_) => BasicValue::F64(F64Value::unitialized()),
            BasicType::Void(_) => BasicValue::Void(VoidValue::unitialized()),
            BasicType::Array(_) => BasicValue::Array(ArrayValue::unitialized()),
            BasicType::Isize(_) => BasicValue::Isize(IsizeValue::unitialized()),
            BasicType::Usize(_) => BasicValue::Usize(UsizeValue::unitialized()),
            BasicType::Struct(_) => BasicValue::Struct(StructValue::unitialized()),
        }
    }
}
