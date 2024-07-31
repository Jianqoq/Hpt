use std::{ cell::RefCell, rc::Rc };

use llvm_sys::core::{
    LLVMBuildAnd,
    LLVMBuildCast,
    LLVMBuildFCmp,
    LLVMBuildFNeg,
    LLVMBuildFPCast,
    LLVMBuildFPToSI,
    LLVMBuildFRem,
    LLVMBuildGEP2,
    LLVMBuildGlobalStringPtr,
    LLVMBuildNeg,
    LLVMBuildNot,
    LLVMBuildOr,
    LLVMBuildPhi,
    LLVMBuildRet,
    LLVMBuildRetVoid,
    LLVMBuildSRem,
    LLVMBuildSelect,
    LLVMBuildStructGEP2,
    LLVMBuildURem,
    LLVMBuildXor,
    LLVMConstReal,
};
use llvm_sys::{
    core::{
        LLVMBuildAdd,
        LLVMBuildAlloca,
        LLVMBuildCall2,
        LLVMBuildFAdd,
        LLVMBuildFDiv,
        LLVMBuildFMul,
        LLVMBuildFSub,
        LLVMBuildICmp,
        LLVMBuildLoad2,
        LLVMBuildMul,
        LLVMBuildSDiv,
        LLVMBuildStore,
        LLVMBuildSub,
        LLVMConstInt,
    },
    prelude::LLVMValueRef,
    LLVMBuilder,
    LLVMIntPredicate,
};
use llvm_sys::{ LLVMOpcode, LLVMRealPredicate };
use crate::context::context::Context;
use crate::types::general_types::GeneralType;
use crate::types::info_trait::UnitizlizeValue;
use crate::types::ptr_type::{ PtrType, StrPtrType };
use crate::types::values::{ BoolValue, PhiValue };
use crate::utils::to_c_str;
use crate::StructType;
use crate::{
    types::{
        block::BasicBlock,
        info_trait::TypeTrait,
        ptr_values::PtrValue,
        values::{ BasicValue, BoolPtrValue, FunctionValue },
    },
    BasicType,
};

#[derive(Debug, PartialEq)]
pub enum PositionState {
    NotSet,
    Set,
}

pub struct Builder {
    pub(crate) builder: *mut LLVMBuilder,
    pub(crate) pos_state: Rc<RefCell<PositionState>>,
}

impl Builder {
    pub fn new(context: &Context) -> Self {
        let builder = unsafe { llvm_sys::core::LLVMCreateBuilderInContext(context.inner()) };
        Builder {
            builder,
            pos_state: Rc::new(RefCell::new(PositionState::NotSet)),
        }
    }
    pub fn get_insert_block(&self) -> BasicBlock {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let block = unsafe { llvm_sys::core::LLVMGetInsertBlock(self.builder) };
        if block.is_null() {
            panic!("Block is null");
        }
        BasicBlock::from(block)
    }

    pub fn build_bitcast(&self, value: BasicValue, cast_type: GeneralType, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMBitCast,
                value.inner(),
                cast_type.inner(),
                name.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_abs(
        &self,
        value_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let zero = self.build_float(value_type, 0.0);
        let is_negative = self.build_float_cmp(
            LLVMRealPredicate::LLVMRealOLT,
            value,
            zero,
            "is_negative"
        );
        let negated = self.build_float_neg(value, "negated");
        self.build_select(is_negative, negated, value, name)
    }

    pub fn build_int_signed_abs(
        &self,
        value_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let zero = self.build_int(value_type, 0, false);
        let is_negative = self.build_int_cmp(
            LLVMIntPredicate::LLVMIntSLT,
            value,
            zero,
            "is_negative"
        );
        let negated = self.build_int_neg(value, "negated");
        self.build_select(is_negative, negated, value, name)
    }

    pub fn build_select(
        &self,
        condition: BasicValue,
        then_value: BasicValue,
        else_value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildSelect(
                self.builder,
                condition.inner(),
                then_value.inner(),
                else_value.inner(),
                name.as_ptr()
            )
        };
        let mut ret = then_value.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_global_string_ptr(&self, value: &str, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(value);
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildGlobalStringPtr(self.builder, c_string.as_ptr(), name.as_ptr())
        };
        let mut ret = StrPtrType::unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_unsigned_int_to_unsigned_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMZExt,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_signed_int_to_signed_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMSExt,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_signed_int_to_unsigned_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMSExt,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_unsigned_int_to_signed_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMZExt,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_ext(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMFPExt,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }
    pub fn build_float_trunc(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMFPTrunc,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_cast(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFPCast(self.builder, value.inner(), cast_type.inner(), c_string.as_ptr())
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_unsigned_int_to_float(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMUIToFP,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_signed_int_to_float(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMSIToFP,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_to_unsigned_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMFPToUI,
                value.inner(),
                cast_type.inner(),
                c_string.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }
    pub fn build_float_to_signed_int(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFPToSI(self.builder, value.inner(), cast_type.inner(), c_string.as_ptr())
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_cmp(
        &self,
        op: LLVMIntPredicate,
        lhs: BasicValue,
        rhs: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildICmp(self.builder, op, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_cmp(
        &self,
        op: LLVMRealPredicate,
        lhs: BasicValue,
        rhs: BasicValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFCmp(self.builder, op, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_unconditional_branch(&self, block: BasicBlock) {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        unsafe {
            llvm_sys::core::LLVMBuildBr(self.builder, block.inner());
        }
    }

    pub fn build_conditional_branch(
        &self,
        condition: BasicValue,
        then_block: BasicBlock,
        else_block: BasicBlock
    ) {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        unsafe {
            llvm_sys::core::LLVMBuildCondBr(
                self.builder,
                condition.inner(),
                then_block.inner(),
                else_block.inner()
            );
        }
    }

    pub fn build_float(&self, float_type: BasicType, value: f64) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let res = unsafe { LLVMConstReal(float_type.inner(), value) };
        let mut ret = float_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int(&self, int_type: BasicType, value: i64, sign_extend: bool) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let res = unsafe { LLVMConstInt(int_type.inner(), value as u64, sign_extend.into()) };
        let mut ret = int_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_alloca<T: TypeTrait>(&self, alloc_type: T, name: &str) -> PtrValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let value = unsafe {
            LLVMBuildAlloca(self.builder, alloc_type.get_type(), c_string.as_ptr())
        };
        PtrValue::Bool(BoolPtrValue::from(value))
    }

    pub fn build_gep<T: TypeTrait>(
        &self,
        pointing_type: T,
        ptr: PtrValue,
        indices: &[BasicValue],
        name: &str
    ) -> PtrValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);

        let mut index_values: Vec<LLVMValueRef> = indices
            .iter()
            .map(|val| val.inner())
            .collect();

        let value = unsafe {
            LLVMBuildGEP2(
                self.builder,
                pointing_type.get_type(),
                ptr.inner(),
                index_values.as_mut_ptr(),
                index_values.len() as u32,
                c_string.as_ptr()
            )
        };
        let mut ret = ptr;
        unsafe {
            ret.set_inner(value);
        }
        ret
    }

    pub fn build_struct_gep(
        &self,
        value: StructType,
        ptr: PtrValue,
        index: u32,
        name: &str
    ) -> PtrValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);

        let value = unsafe {
            LLVMBuildStructGEP2(self.builder, value.inner(), ptr.inner(), index, c_string.as_ptr())
        };
        let mut ret = ptr;
        unsafe {
            ret.set_inner(value);
        }
        ret
    }

    pub fn build_int_neg(&self, value: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe { LLVMBuildNeg(self.builder, value.inner(), c_string.as_ptr()) };
        let mut ret = value.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_store_ptr(&self, ptr: PtrValue, value: PtrValue) {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        unsafe {
            LLVMBuildStore(self.builder, value.inner(), ptr.inner());
        }
    }

    pub fn build_store(&self, ptr: PtrValue, value: BasicValue) {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        unsafe {
            LLVMBuildStore(self.builder, value.inner(), ptr.inner());
        }
    }

    pub fn build_load<T: TypeTrait + UnitizlizeValue>(
        &self,
        pointing_type: T,
        ptr: PtrValue,
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildLoad2(self.builder, pointing_type.get_type(), ptr.inner(), name.as_ptr())
        };
        let mut ret = pointing_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_ptr_bitcast(&self, cast_type: PtrType, value: PtrValue, name: &str) -> PtrValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMBitCast,
                value.inner(),
                cast_type.inner(),
                name.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret.to_ptr_value()
    }

    pub fn build_val_bitcast(
        &self,
        cast_type: BasicType,
        value: BasicValue,
        name: &str
    ) -> PtrValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe {
            LLVMBuildCast(
                self.builder,
                LLVMOpcode::LLVMBitCast,
                value.inner(),
                cast_type.inner(),
                name.as_ptr()
            )
        };
        let mut ret = cast_type.unitialize();
        unsafe {
            ret.set_inner(res);
        }
        ret.to_ptr_value()
    }

    pub fn build_call(
        &self,
        fn_value: &FunctionValue,
        args: &[BasicValue],
        name: &str
    ) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let mut args: Vec<LLVMValueRef> = args
            .iter()
            .map(|val| val.inner())
            .collect();
        let res = unsafe {
            LLVMBuildCall2(
                self.builder,
                fn_value.to_type().inner(),
                fn_value.inner(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_string.as_ptr()
            )
        };
        let mut ret = fn_value.empty_value();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_phi(&self, ty: BasicType, name: &str) -> PhiValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let name = to_c_str(name);
        let res = unsafe { LLVMBuildPhi(self.builder, ty.inner(), name.as_ptr()) };
        PhiValue::new(res, ty)
    }

    pub fn build_float_add(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFAdd(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_add(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildAdd(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_neg(&self, value: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe { LLVMBuildFNeg(self.builder, value.inner(), c_string.as_ptr()) };
        let mut ret = value.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_sub(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFSub(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_sub(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildSub(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_mul(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFMul(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_and(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildAnd(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        BoolValue::from(res).into()
    }

    pub fn build_int_or(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe { LLVMBuildOr(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr()) };
        BoolValue::from(res).into()
    }

    pub fn build_int_xor(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildXor(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        BoolValue::from(res).into()
    }

    pub fn build_int_not(&self, value: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe { LLVMBuildNot(self.builder, value.inner(), c_string.as_ptr()) };
        BoolValue::from(res).into()
    }

    pub fn build_int_mul(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildMul(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_uint_rem(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildURem(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_rem(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildSRem(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_div(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFDiv(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_float_rem(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildFRem(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn build_int_signed_div(&self, lhs: BasicValue, rhs: BasicValue, name: &str) -> BasicValue {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let c_string = to_c_str(name);
        let res = unsafe {
            LLVMBuildSDiv(self.builder, lhs.inner(), rhs.inner(), c_string.as_ptr())
        };
        let mut ret = lhs.unitialized();
        unsafe {
            ret.set_inner(res);
        }
        ret
    }

    pub fn position_at_end(&self, block: BasicBlock) {
        unsafe {
            llvm_sys::core::LLVMPositionBuilderAtEnd(self.builder, block.inner());
        }
        *self.pos_state.borrow_mut() = PositionState::Set;
    }

    pub fn build_return(&self, value: Option<BasicValue>) {
        if *self.pos_state.borrow() == PositionState::NotSet {
            panic!("Position not set");
        }
        let value = unsafe {
            value.map_or_else(
                || LLVMBuildRetVoid(self.builder),
                |value| LLVMBuildRet(self.builder, value.inner())
            )
        };
        assert!(!value.is_null());
    }
}
