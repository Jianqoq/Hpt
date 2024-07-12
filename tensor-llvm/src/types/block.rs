use llvm_sys::prelude::LLVMBasicBlockRef;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BasicBlock {
    pub(crate) block: LLVMBasicBlockRef,
}

impl From<LLVMBasicBlockRef> for BasicBlock {
    fn from(block: LLVMBasicBlockRef) -> Self {
        assert!(
            !block.is_null(),
            "BasicBlock::from(LLVMBasicBlockRef) was called with a null pointer"
        );
        Self { block }
    }
}

impl BasicBlock {
    pub fn inner(&self) -> LLVMBasicBlockRef {
        self.block
    }
}
