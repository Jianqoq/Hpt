use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
};

pub struct CodeGen {
    ctx: Context,
    module: tensor_llvm::module::module::Module,
    builder: Builder,
}

impl CodeGen {
    pub fn new(
        ctx: Context,
        module: tensor_llvm::module::module::Module,
        builder: Builder
    ) -> Self {
        CodeGen {
            ctx,
            module,
            builder,
        }
    }
}
