use std::rc::Rc;

use super::block::BasicBlock;
use super::general_types::GeneralType;
use super::ptr_values::PtrValue;
use crate::builder::builder::Builder;
use crate::context::context::Context;
use crate::types::ptr_type::*;
use crate::types::types::*;
use llvm_sys::prelude::LLVMBasicBlockRef;
use llvm_sys::prelude::LLVMValueRef;
use paste::paste;
use tensor_types::dtype::Dtype;
macro_rules! register_val_type {
    ($base_name:ident) => {
        paste! {
            #[derive(Debug, PartialEq, Copy, Clone, Eq, Hash)]
            pub struct  [<$base_name Value>]  {
                pub(crate) value: LLVMValueRef,
            }

            impl From<LLVMValueRef> for [<$base_name Value>] {
                fn from(value: LLVMValueRef) -> Self {
                    if value.is_null() {
                        panic!("Value is null");
                    }
                    Self { value }
                }
            }

            impl [<$base_name Value>] {
                pub fn inner(&self) -> LLVMValueRef {
                    self.value
                }

                pub fn unitialzed() -> Self {
                    Self {
                        value: std::ptr::null_mut(),
                    }
                }

                pub fn to_type(&self) -> [<$base_name Type>] {
                    let return_type = unsafe {llvm_sys::core::LLVMGlobalGetValueType(self.value)};
                    assert!(!return_type.is_null());
                    [<$base_name Type>]::from(return_type)
                }

                pub fn unitialized() -> Self {
                    Self {
                        value: std::ptr::null_mut(),
                    }
                }
            }
        }
    };
}

register_val_type!(Bool);
register_val_type!(I8);
register_val_type!(U8);
register_val_type!(I16);
register_val_type!(U16);
register_val_type!(I32);
register_val_type!(U32);
register_val_type!(I64);
register_val_type!(U64);
register_val_type!(BF16);
register_val_type!(F16);
register_val_type!(F32);
register_val_type!(F64);
register_val_type!(Void);
register_val_type!(Isize);
register_val_type!(BoolPtr);
register_val_type!(I8Ptr);
register_val_type!(U8Ptr);
register_val_type!(I16Ptr);
register_val_type!(U16Ptr);
register_val_type!(I32Ptr);
register_val_type!(U32Ptr);
register_val_type!(I64Ptr);
register_val_type!(U64Ptr);
register_val_type!(BF16Ptr);
register_val_type!(F16Ptr);
register_val_type!(F32Ptr);
register_val_type!(F64Ptr);
register_val_type!(VoidPtr);
register_val_type!(IsizePtr);
register_val_type!(ArrayPtr);
register_val_type!(FunctionPtr);
register_val_type!(StrPtr);
register_val_type!(Array);
register_val_type!(Str);
register_val_type!(Struct);
register_val_type!(StructPtr);

#[derive(Debug, PartialEq, Copy, Clone, Eq, Hash)]
pub enum BasicValue {
    Bool(BoolValue),
    I8(I8Value),
    U8(U8Value),
    I16(I16Value),
    U16(U16Value),
    I32(I32Value),
    U32(U32Value),
    I64(I64Value),
    U64(U64Value),
    Isize(IsizeValue),
    BF16(BF16Value),
    F16(F16Value),
    F32(F32Value),
    F64(F64Value),
    Void(VoidValue),
    Array(ArrayValue),
    Str(StrValue),
    Phi(PhiValue),
    StrPtr(StrPtrValue),
    BoolPtr(BoolPtrValue),
    I8Ptr(I8PtrValue),
    U8Ptr(U8PtrValue),
    I16Ptr(I16PtrValue),
    U16Ptr(U16PtrValue),
    I32Ptr(I32PtrValue),
    U32Ptr(U32PtrValue),
    I64Ptr(I64PtrValue),
    U64Ptr(U64PtrValue),
    BF16Ptr(BF16PtrValue),
    F16Ptr(F16PtrValue),
    F32Ptr(F32PtrValue),
    F64Ptr(F64PtrValue),
    VoidPtr(VoidPtrValue),
    IsizePtr(IsizePtrValue),
    ArrayPtr(ArrayPtrValue),
    FunctionPtr(FunctionPtrValue),
    Struct(StructValue),
    StructPtr(StructPtrValue),
    None,
}

impl BasicValue {
    pub fn unitialized(&self) -> Self {
        match self {
            BasicValue::Bool(_) => BasicValue::Bool(BoolValue::unitialized()),
            BasicValue::I8(_) => BasicValue::I8(I8Value::unitialized()),
            BasicValue::U8(_) => BasicValue::U8(U8Value::unitialized()),
            BasicValue::I16(_) => BasicValue::I16(I16Value::unitialized()),
            BasicValue::U16(_) => BasicValue::U16(U16Value::unitialized()),
            BasicValue::I32(_) => BasicValue::I32(I32Value::unitialized()),
            BasicValue::U32(_) => BasicValue::U32(U32Value::unitialized()),
            BasicValue::I64(_) => BasicValue::I64(I64Value::unitialized()),
            BasicValue::U64(_) => BasicValue::U64(U64Value::unitialized()),
            BasicValue::BF16(_) => BasicValue::BF16(BF16Value::unitialized()),
            BasicValue::F16(_) => BasicValue::F16(F16Value::unitialized()),
            BasicValue::F32(_) => BasicValue::F32(F32Value::unitialized()),
            BasicValue::F64(_) => BasicValue::F64(F64Value::unitialized()),
            BasicValue::Void(_) => BasicValue::Void(VoidValue::unitialized()),
            BasicValue::Isize(_) => BasicValue::Isize(IsizeValue::unitialized()),
            BasicValue::Array(_) => BasicValue::Array(ArrayValue::unitialized()),
            BasicValue::BoolPtr(_) => BasicValue::BoolPtr(BoolPtrValue::unitialized()),
            BasicValue::I8Ptr(_) => BasicValue::I8Ptr(I8PtrValue::unitialized()),
            BasicValue::U8Ptr(_) => BasicValue::U8Ptr(U8PtrValue::unitialized()),
            BasicValue::I16Ptr(_) => BasicValue::I16Ptr(I16PtrValue::unitialized()),
            BasicValue::U16Ptr(_) => BasicValue::U16Ptr(U16PtrValue::unitialized()),
            BasicValue::I32Ptr(_) => BasicValue::I32Ptr(I32PtrValue::unitialized()),
            BasicValue::U32Ptr(_) => BasicValue::U32Ptr(U32PtrValue::unitialized()),
            BasicValue::I64Ptr(_) => BasicValue::I64Ptr(I64PtrValue::unitialized()),
            BasicValue::U64Ptr(_) => BasicValue::U64Ptr(U64PtrValue::unitialized()),
            BasicValue::BF16Ptr(_) => BasicValue::BF16Ptr(BF16PtrValue::unitialized()),
            BasicValue::F16Ptr(_) => BasicValue::F16Ptr(F16PtrValue::unitialized()),
            BasicValue::F32Ptr(_) => BasicValue::F32Ptr(F32PtrValue::unitialized()),
            BasicValue::F64Ptr(_) => BasicValue::F64Ptr(F64PtrValue::unitialized()),
            BasicValue::VoidPtr(_) => BasicValue::VoidPtr(VoidPtrValue::unitialized()),
            BasicValue::IsizePtr(_) => BasicValue::IsizePtr(IsizePtrValue::unitialized()),
            BasicValue::ArrayPtr(_) => BasicValue::ArrayPtr(ArrayPtrValue::unitialized()),
            BasicValue::FunctionPtr(_) => BasicValue::FunctionPtr(FunctionPtrValue::unitialized()),
            BasicValue::Str(_) => BasicValue::Str(StrValue::unitialized()),
            BasicValue::StrPtr(_) => BasicValue::StrPtr(StrPtrValue::unitialized()),
            BasicValue::Phi(_) => BasicValue::Phi(PhiValue::new(std::ptr::null_mut())),
            BasicValue::Struct(_) => BasicValue::Struct(StructValue::unitialized()),
            BasicValue::StructPtr(_) => BasicValue::StructPtr(StructPtrValue::unitialized()),
            BasicValue::None => BasicValue::None,
        }
    }

    pub fn to_basic_type(&self) -> BasicType {
        match self {
            BasicValue::Bool(val) => BasicType::Bool(val.to_type()),
            BasicValue::I8(val) => BasicType::I8(val.to_type()),
            BasicValue::U8(val) => BasicType::U8(val.to_type()),
            BasicValue::I16(val) => BasicType::I16(val.to_type()),
            BasicValue::U16(val) => BasicType::U16(val.to_type()),
            BasicValue::I32(val) => BasicType::I32(val.to_type()),
            BasicValue::U32(val) => BasicType::U32(val.to_type()),
            BasicValue::I64(val) => BasicType::I64(val.to_type()),
            BasicValue::U64(val) => BasicType::U64(val.to_type()),
            BasicValue::BF16(val) => BasicType::BF16(val.to_type()),
            BasicValue::F16(val) => BasicType::F16(val.to_type()),
            BasicValue::F32(val) => BasicType::F32(val.to_type()),
            BasicValue::F64(val) => BasicType::F64(val.to_type()),
            BasicValue::Void(val) => BasicType::Void(val.to_type()),
            BasicValue::Isize(val) => BasicType::Isize(val.to_type()),
            BasicValue::Array(val) => BasicType::Array(val.to_type()),
            _ => panic!("Type not supported, {:?}", self),
        }
    }

    pub fn to_general_type(&self) -> GeneralType {
        match self {
            BasicValue::Bool(val) => GeneralType::Bool(val.to_type()),
            BasicValue::I8(val) => GeneralType::I8(val.to_type()),
            BasicValue::U8(val) => GeneralType::U8(val.to_type()),
            BasicValue::I16(val) => GeneralType::I16(val.to_type()),
            BasicValue::U16(val) => GeneralType::U16(val.to_type()),
            BasicValue::I32(val) => GeneralType::I32(val.to_type()),
            BasicValue::U32(val) => GeneralType::U32(val.to_type()),
            BasicValue::I64(val) => GeneralType::I64(val.to_type()),
            BasicValue::U64(val) => GeneralType::U64(val.to_type()),
            BasicValue::BF16(val) => GeneralType::BF16(val.to_type()),
            BasicValue::F16(val) => GeneralType::F16(val.to_type()),
            BasicValue::F32(val) => GeneralType::F32(val.to_type()),
            BasicValue::F64(val) => GeneralType::F64(val.to_type()),
            BasicValue::Void(val) => GeneralType::Void(val.to_type()),
            BasicValue::Isize(val) => GeneralType::Isize(val.to_type()),
            BasicValue::Array(val) => GeneralType::Array(val.to_type()),
            _ => panic!("Type not supported, {:?}", self),
        }
    }

    pub fn to_dtype(&self) -> Dtype {
        match self {
            BasicValue::Bool(_) => Dtype::Bool,
            BasicValue::I8(_) => Dtype::I8,
            BasicValue::U8(_) => Dtype::U8,
            BasicValue::I16(_) => Dtype::I16,
            BasicValue::U16(_) => Dtype::U16,
            BasicValue::I32(_) => Dtype::I32,
            BasicValue::U32(_) => Dtype::U32,
            BasicValue::I64(_) => Dtype::I64,
            BasicValue::U64(_) => Dtype::U64,
            BasicValue::BF16(_) => Dtype::BF16,
            BasicValue::F16(_) => Dtype::F16,
            BasicValue::F32(_) => Dtype::F32,
            BasicValue::F64(_) => Dtype::F64,
            BasicValue::Isize(_) => Dtype::Isize,
            _ => panic!("Type not supported, {:?}", self),
        }
    }
    pub fn cast(
        &self,
        to_cast: Dtype,
        var_name: &str,
        context: &Context,
        builder: &Builder
    ) -> BasicValue {
        let res_type = dtype_to_llvm(to_cast, context);
        match self {
            BasicValue::F64(_) | BasicValue::F32(_) =>
                match to_cast {
                    Dtype::F64 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::F32 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::F16 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::Bool => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I8 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U8 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I16 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U16 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I32 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U32 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I64 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U64 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    _ => panic!("Unsupported cast"),
                }
            BasicValue::I64(_) | BasicValue::I32(_) | BasicValue::I16(_) | BasicValue::I8(_) => {
                match to_cast {
                    Dtype::F64 => builder.build_signed_int_to_float(res_type, *self, var_name),
                    Dtype::F32 => builder.build_signed_int_to_float(res_type, *self, var_name),
                    Dtype::Bool =>
                        builder.build_signed_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I8 => builder.build_signed_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U8 =>
                        builder.build_signed_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I16 => builder.build_signed_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U16 =>
                        builder.build_signed_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I32 => builder.build_signed_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U32 =>
                        builder.build_signed_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I64 => builder.build_signed_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U64 =>
                        builder.build_signed_int_to_unsigned_int(res_type, *self, var_name),
                    _ => panic!("Unsupported cast"),
                }
            }
            BasicValue::U8(_) | BasicValue::U16(_) | BasicValue::U32(_) | BasicValue::U64(_) => {
                match to_cast {
                    Dtype::F64 => builder.build_unsigned_int_to_float(res_type, *self, var_name),
                    Dtype::F32 => builder.build_unsigned_int_to_float(res_type, *self, var_name),
                    Dtype::Bool =>
                        builder.build_unsigned_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I8 =>
                        builder.build_unsigned_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U8 =>
                        builder.build_unsigned_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I16 =>
                        builder.build_unsigned_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U16 =>
                        builder.build_unsigned_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I32 =>
                        builder.build_unsigned_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U32 =>
                        builder.build_unsigned_int_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I64 =>
                        builder.build_unsigned_int_to_signed_int(res_type, *self, var_name),
                    Dtype::U64 =>
                        builder.build_unsigned_int_to_unsigned_int(res_type, *self, var_name),
                    _ => panic!("Unsupported cast"),
                }
            }
            BasicValue::F16(_) =>
                match to_cast {
                    Dtype::F64 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::F32 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::F16 => builder.build_float_cast(res_type, *self, var_name),
                    Dtype::Bool => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I8 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U8 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I16 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U16 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I32 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U32 => builder.build_float_to_unsigned_int(res_type, *self, var_name),
                    Dtype::I64 => builder.build_float_to_signed_int(res_type, *self, var_name),
                    Dtype::U64 => builder.build_float_to_unsigned_int(res_type, *self, var_name),

                    _ => panic!("Unsupported cast"),
                }
            _ => panic!("{}", &format!("Unsupported cast {:?}", *self)),
        }
    }
}

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

macro_rules! impl_from {
    ($base_name:ident) => {
        paste! {impl From<[<$base_name Value>]> for BasicValue {
                fn from(value: [<$base_name Value>]) -> Self {
                    BasicValue::$base_name(value)
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
impl_from!(Isize);
impl_from!(Array);
impl_from!(BoolPtr);
impl_from!(I8Ptr);
impl_from!(U8Ptr);
impl_from!(I16Ptr);
impl_from!(U16Ptr);
impl_from!(I32Ptr);
impl_from!(U32Ptr);
impl_from!(I64Ptr);
impl_from!(U64Ptr);
impl_from!(BF16Ptr);
impl_from!(F16Ptr);
impl_from!(F32Ptr);
impl_from!(F64Ptr);
impl_from!(VoidPtr);
impl_from!(IsizePtr);
impl_from!(ArrayPtr);
impl_from!(FunctionPtr);
impl_from!(Phi);

impl BasicValue {
    pub fn inner(&self) -> LLVMValueRef {
        match self {
            BasicValue::Bool(val) => val.inner(),
            BasicValue::I8(val) => val.inner(),
            BasicValue::U8(val) => val.inner(),
            BasicValue::I16(val) => val.inner(),
            BasicValue::U16(val) => val.inner(),
            BasicValue::I32(val) => val.inner(),
            BasicValue::U32(val) => val.inner(),
            BasicValue::I64(val) => val.inner(),
            BasicValue::U64(val) => val.inner(),
            BasicValue::BF16(val) => val.inner(),
            BasicValue::F16(val) => val.inner(),
            BasicValue::F32(val) => val.inner(),
            BasicValue::F64(val) => val.inner(),
            BasicValue::Void(val) => val.inner(),
            BasicValue::BoolPtr(val) => val.inner(),
            BasicValue::I8Ptr(val) => val.inner(),
            BasicValue::U8Ptr(val) => val.inner(),
            BasicValue::I16Ptr(val) => val.inner(),
            BasicValue::U16Ptr(val) => val.inner(),
            BasicValue::I32Ptr(val) => val.inner(),
            BasicValue::U32Ptr(val) => val.inner(),
            BasicValue::I64Ptr(val) => val.inner(),
            BasicValue::U64Ptr(val) => val.inner(),
            BasicValue::BF16Ptr(val) => val.inner(),
            BasicValue::F16Ptr(val) => val.inner(),
            BasicValue::F32Ptr(val) => val.inner(),
            BasicValue::F64Ptr(val) => val.inner(),
            BasicValue::VoidPtr(val) => val.inner(),
            BasicValue::Isize(val) => val.inner(),
            BasicValue::IsizePtr(val) => val.inner(),
            BasicValue::ArrayPtr(val) => val.inner(),
            BasicValue::FunctionPtr(val) => val.inner(),
            BasicValue::Array(val) => val.inner(),
            BasicValue::Str(val) => val.inner(),
            BasicValue::StrPtr(val) => val.inner(),
            BasicValue::Phi(val) => val.inner(),
            BasicValue::None => std::ptr::null_mut(),
            BasicValue::Struct(val) => val.inner(),
            BasicValue::StructPtr(val) => val.inner(),
        }
    }
}

macro_rules! impl_to {
    ($name:ident, $enum_name:ident) => {
        paste! {
            pub fn $name(&self) -> [<$enum_name Value>] {
                match self {
                    BasicValue::$enum_name(val) => *val,
                    _ => panic!("Value is not a {}", stringify!([<$enum_name Value>])),
                }
            }
        }
    };
}

impl BasicValue {
    impl_to!(to_bool, Bool);
    impl_to!(to_i8, I8);
    impl_to!(to_u8, U8);
    impl_to!(to_i16, I16);
    impl_to!(to_u16, U16);
    impl_to!(to_i32, I32);
    impl_to!(to_u32, U32);
    impl_to!(to_i64, I64);
    impl_to!(to_u64, U64);
    impl_to!(to_bf16, BF16);
    impl_to!(to_f16, F16);
    impl_to!(to_f32, F32);
    impl_to!(to_f64, F64);
    impl_to!(to_void, Void);
    impl_to!(to_bool_ptr, BoolPtr);
    impl_to!(to_i8_ptr, I8Ptr);
    impl_to!(to_u8_ptr, U8Ptr);
    impl_to!(to_i16_ptr, I16Ptr);
    impl_to!(to_u16_ptr, U16Ptr);
    impl_to!(to_i32_ptr, I32Ptr);
    impl_to!(to_u32_ptr, U32Ptr);
    impl_to!(to_i64_ptr, I64Ptr);
    impl_to!(to_u64_ptr, U64Ptr);
    impl_to!(to_bf16_ptr, BF16Ptr);
    impl_to!(to_f16_ptr, F16Ptr);
    impl_to!(to_f32_ptr, F32Ptr);
    impl_to!(to_f64_ptr, F64Ptr);
    impl_to!(to_void_ptr, VoidPtr);
    impl_to!(to_str_ptr, StrPtr);
    impl_to!(to_str, Str);

    pub fn to_ptr_value(&self) -> PtrValue {
        PtrValue::from(*self)
    }
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
pub struct FunctionValue {
    pub(crate) value: LLVMValueRef,
    pub(crate) ret_type: GeneralType,
    pub(crate) params: Rc<Vec<GeneralType>>,
    pub(crate) num_params: usize,
}

impl FunctionValue {
    pub fn new(
        value: LLVMValueRef,
        ret_type: GeneralType,
        params: Rc<Vec<GeneralType>>,
        num_params: usize
    ) -> Self {
        Self {
            value,
            ret_type,
            params,
            num_params,
        }
    }

    pub fn ret_type(&self) -> &GeneralType {
        &self.ret_type
    }

    pub fn empty_value(&self) -> BasicValue {
        match &self.ret_type {
            GeneralType::Bool(_) => BoolType::unitialize(),
            GeneralType::I8(_) => I8Type::unitialize(),
            GeneralType::U8(_) => U8Type::unitialize(),
            GeneralType::I16(_) => I16Type::unitialize(),
            GeneralType::U16(_) => U16Type::unitialize(),
            GeneralType::I32(_) => I32Type::unitialize(),
            GeneralType::U32(_) => U32Type::unitialize(),
            GeneralType::I64(_) => I64Type::unitialize(),
            GeneralType::U64(_) => U64Type::unitialize(),
            GeneralType::BF16(_) => BF16Type::unitialize(),
            GeneralType::F16(_) => F16Type::unitialize(),
            GeneralType::F32(_) => F32Type::unitialize(),
            GeneralType::F64(_) => F64Type::unitialize(),
            GeneralType::Void(_) => VoidType::unitialize(),
            GeneralType::Isize(_) => IsizeType::unitialize(),
            GeneralType::Array(_) => ArrayType::unitialize(),
            GeneralType::BoolPtr(_) => BoolPtrType::unitialize(),
            GeneralType::I8Ptr(_) => I8PtrType::unitialize(),
            GeneralType::U8Ptr(_) => U8PtrType::unitialize(),
            GeneralType::I16Ptr(_) => I16PtrType::unitialize(),
            GeneralType::I64Ptr(_) => I64PtrType::unitialize(),
            GeneralType::U16Ptr(_) => U16PtrType::unitialize(),
            GeneralType::I32Ptr(_) => I32PtrType::unitialize(),
            GeneralType::U32Ptr(_) => U32PtrType::unitialize(),
            GeneralType::U64Ptr(_) => U64PtrType::unitialize(),
            GeneralType::BF16Ptr(_) => BF16PtrType::unitialize(),
            GeneralType::F16Ptr(_) => F16PtrType::unitialize(),
            GeneralType::F32Ptr(_) => F32PtrType::unitialize(),
            GeneralType::F64Ptr(_) => F64PtrType::unitialize(),
            GeneralType::VoidPtr(_) => VoidPtrType::unitialize(),
            GeneralType::IsizePtr(_) => IsizePtrType::unitialize(),
            GeneralType::ArrayPtr(_) => ArrayPtrType::unitialize(),
            GeneralType::FunctionPtr(_) => FunctionPtrType::unitialize(),
            GeneralType::Function(_) => panic!("Function type not supported"),
            GeneralType::Str(_) => StrType::unitialize(),
            GeneralType::StrPtr(_) => StrPtrType::unitialize(),
            GeneralType::Struct(_) => StructType::unitialize(),
            GeneralType::StructPtr(_) => StructPtrType::unitialize(),
        }
    }

    pub fn value(&self) -> LLVMValueRef {
        self.value
    }

    pub fn inner(&self) -> LLVMValueRef {
        self.value
    }
    pub fn to_type(&self) -> FunctionType {
        let return_type = unsafe { llvm_sys::core::LLVMGlobalGetValueType(self.value) };
        assert!(!return_type.is_null());
        FunctionType::new(return_type, self.ret_type.clone(), self.params.clone(), self.num_params)
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub struct PhiValue {
    pub(crate) value: LLVMValueRef,
}

impl PhiValue {
    pub fn new(value: LLVMValueRef) -> Self {
        Self {
            value,
        }
    }

    pub fn inner(&self) -> LLVMValueRef {
        self.value
    }

    pub fn add_incoming(&self, incoming: &[(&BasicValue, BasicBlock)]) {
        let (mut values, mut basic_blocks): (Vec<LLVMValueRef>, Vec<LLVMBasicBlockRef>) = {
            incoming
                .iter()
                .map(|&(v, bb)| (v.inner(), bb.inner()))
                .unzip()
        };

        unsafe {
            llvm_sys::core::LLVMAddIncoming(
                self.value,
                values.as_mut_ptr(),
                basic_blocks.as_mut_ptr(),
                incoming.len() as u32
            );
        }
    }
}

impl FunctionValue {
    pub fn get_nth_param(&self, index: usize) -> BasicValue {
        if index >= self.num_params {
            panic!("Index out of bounds");
        }
        let param = unsafe { llvm_sys::core::LLVMGetParam(self.value, index as u32) };
        let types = &self.params[index];
        match types {
            GeneralType::Bool(_) => BasicValue::Bool(BoolValue::from(param)),
            GeneralType::I8(_) => BasicValue::I8(I8Value::from(param)),
            GeneralType::U8(_) => BasicValue::U8(U8Value::from(param)),
            GeneralType::I16(_) => BasicValue::I16(I16Value::from(param)),
            GeneralType::U16(_) => BasicValue::U16(U16Value::from(param)),
            GeneralType::I32(_) => BasicValue::I32(I32Value::from(param)),
            GeneralType::U32(_) => BasicValue::U32(U32Value::from(param)),
            GeneralType::I64(_) => BasicValue::I64(I64Value::from(param)),
            GeneralType::U64(_) => BasicValue::U64(U64Value::from(param)),
            GeneralType::BF16(_) => BasicValue::BF16(BF16Value::from(param)),
            GeneralType::F16(_) => BasicValue::F16(F16Value::from(param)),
            GeneralType::F32(_) => BasicValue::F32(F32Value::from(param)),
            GeneralType::F64(_) => BasicValue::F64(F64Value::from(param)),
            GeneralType::Void(_) => BasicValue::Void(VoidValue::from(param)),
            GeneralType::Isize(_) => BasicValue::Isize(IsizeValue::from(param)),
            GeneralType::Array(_) => BasicValue::Array(ArrayValue::from(param)),
            GeneralType::BoolPtr(_) => BasicValue::BoolPtr(BoolPtrValue::from(param)),
            GeneralType::I8Ptr(_) => BasicValue::I8Ptr(I8PtrValue::from(param)),
            GeneralType::U8Ptr(_) => BasicValue::U8Ptr(U8PtrValue::from(param)),
            GeneralType::I16Ptr(_) => BasicValue::I16Ptr(I16PtrValue::from(param)),
            GeneralType::U16Ptr(_) => BasicValue::U16Ptr(U16PtrValue::from(param)),
            GeneralType::I32Ptr(_) => BasicValue::I32Ptr(I32PtrValue::from(param)),
            GeneralType::U32Ptr(_) => BasicValue::U32Ptr(U32PtrValue::from(param)),
            GeneralType::I64Ptr(_) => BasicValue::I64Ptr(I64PtrValue::from(param)),
            GeneralType::U64Ptr(_) => BasicValue::U64Ptr(U64PtrValue::from(param)),
            GeneralType::BF16Ptr(_) => BasicValue::BF16Ptr(BF16PtrValue::from(param)),
            GeneralType::F16Ptr(_) => BasicValue::F16Ptr(F16PtrValue::from(param)),
            GeneralType::F32Ptr(_) => BasicValue::F32Ptr(F32PtrValue::from(param)),
            GeneralType::F64Ptr(_) => BasicValue::F64Ptr(F64PtrValue::from(param)),
            GeneralType::VoidPtr(_) => BasicValue::VoidPtr(VoidPtrValue::from(param)),
            GeneralType::IsizePtr(_) => BasicValue::IsizePtr(IsizePtrValue::from(param)),
            GeneralType::ArrayPtr(_) => BasicValue::ArrayPtr(ArrayPtrValue::from(param)),
            GeneralType::FunctionPtr(_) => BasicValue::FunctionPtr(FunctionPtrValue::from(param)),
            GeneralType::Function(_) => todo!(),
            GeneralType::Str(_) => BasicValue::Str(StrValue::from(param)),
            GeneralType::StrPtr(_) => BasicValue::StrPtr(StrPtrValue::from(param)),
            GeneralType::Struct(_) => BasicValue::Struct(StructValue::from(param)),
            GeneralType::StructPtr(_) => BasicValue::StructPtr(StructPtrValue::from(param)),
        }
    }
}
impl BasicValue {
    pub unsafe fn set_inner(&mut self, value: LLVMValueRef) {
        assert!(!value.is_null());
        match self {
            BasicValue::Bool(val) => {
                val.value = value;
            }
            BasicValue::I8(val) => {
                val.value = value;
            }
            BasicValue::U8(val) => {
                val.value = value;
            }
            BasicValue::I16(val) => {
                val.value = value;
            }
            BasicValue::U16(val) => {
                val.value = value;
            }
            BasicValue::I32(val) => {
                val.value = value;
            }
            BasicValue::U32(val) => {
                val.value = value;
            }
            BasicValue::I64(val) => {
                val.value = value;
            }
            BasicValue::U64(val) => {
                val.value = value;
            }
            BasicValue::BF16(val) => {
                val.value = value;
            }
            BasicValue::F16(val) => {
                val.value = value;
            }
            BasicValue::F32(val) => {
                val.value = value;
            }
            BasicValue::F64(val) => {
                val.value = value;
            }
            BasicValue::Void(val) => {
                val.value = value;
            }
            BasicValue::BoolPtr(val) => {
                val.value = value;
            }
            BasicValue::I8Ptr(val) => {
                val.value = value;
            }
            BasicValue::U8Ptr(val) => {
                val.value = value;
            }
            BasicValue::I16Ptr(val) => {
                val.value = value;
            }
            BasicValue::U16Ptr(val) => {
                val.value = value;
            }
            BasicValue::I32Ptr(val) => {
                val.value = value;
            }
            BasicValue::U32Ptr(val) => {
                val.value = value;
            }
            BasicValue::I64Ptr(val) => {
                val.value = value;
            }
            BasicValue::U64Ptr(val) => {
                val.value = value;
            }
            BasicValue::BF16Ptr(val) => {
                val.value = value;
            }
            BasicValue::F16Ptr(val) => {
                val.value = value;
            }
            BasicValue::F32Ptr(val) => {
                val.value = value;
            }
            BasicValue::F64Ptr(val) => {
                val.value = value;
            }
            BasicValue::VoidPtr(val) => {
                val.value = value;
            }
            BasicValue::Isize(val) => {
                val.value = value;
            }
            BasicValue::IsizePtr(val) => {
                val.value = value;
            }
            BasicValue::ArrayPtr(val) => {
                val.value = value;
            }
            BasicValue::FunctionPtr(val) => {
                val.value = value;
            }
            BasicValue::Array(val) => {
                val.value = value;
            }
            BasicValue::Str(val) => {
                val.value = value;
            }
            BasicValue::StrPtr(val) => {
                val.value = value;
            }
            BasicValue::Phi(val) => {
                val.value = value;
            }
            BasicValue::Struct(val) => {
                val.value = value;
            }
            BasicValue::StructPtr(val) => {
                val.value = value;
            }
            BasicValue::None => {}
        }
    }

    pub fn has_sign(&self) -> bool {
        match self {
            BasicValue::Bool(_) => false,
            BasicValue::I8(_) => true,
            BasicValue::U8(_) => false,
            BasicValue::I16(_) => true,
            BasicValue::U16(_) => false,
            BasicValue::I32(_) => true,
            BasicValue::U32(_) => false,
            BasicValue::I64(_) => true,
            BasicValue::U64(_) => false,
            BasicValue::BF16(_) => true,
            BasicValue::F16(_) => true,
            BasicValue::F32(_) => true,
            BasicValue::F64(_) => true,
            BasicValue::Void(_) => false,
            BasicValue::Isize(_) => true,
            BasicValue::Array(_) => false,
            BasicValue::BoolPtr(_) => false,
            BasicValue::I8Ptr(_) => false,
            BasicValue::U8Ptr(_) => false,
            BasicValue::I16Ptr(_) => false,
            BasicValue::U16Ptr(_) => false,
            BasicValue::I32Ptr(_) => false,
            BasicValue::U32Ptr(_) => false,
            BasicValue::I64Ptr(_) => false,
            BasicValue::U64Ptr(_) => false,
            BasicValue::BF16Ptr(_) => false,
            BasicValue::F16Ptr(_) => false,
            BasicValue::F32Ptr(_) => false,
            BasicValue::F64Ptr(_) => false,
            BasicValue::VoidPtr(_) => false,
            BasicValue::IsizePtr(_) => false,
            BasicValue::ArrayPtr(_) => false,
            BasicValue::FunctionPtr(_) => false,
            BasicValue::Str(_) => false,
            BasicValue::StrPtr(_) => false,
            BasicValue::Phi(_) => false,
            BasicValue::Struct(_) => false,
            BasicValue::StructPtr(_) => false,
            BasicValue::None => false,
        }
    }
}
