use llvm_sys::prelude::LLVMTypeRef;

use super::values::BasicValue;

pub trait TypeTrait {
    fn get_type(&self) -> LLVMTypeRef;
}

pub trait UnitizlizeValue {
    fn unitialize(&self) -> BasicValue;
}
