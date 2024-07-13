use std::{ cell::RefCell, rc::Rc };

use tensor_llvm::{ builder::builder::{ Builder, PositionState }, context::context::Context };

use crate::halide::module::Module;

pub struct CodeGen {
    ctx: Context,
    module: tensor_llvm::module::module::Module,
    builder: Builder,
}

impl CodeGen {
    pub fn new(ctx: Context, module: &Module, builder: Builder) -> Self {
        let _module = tensor_llvm::module::module::Module::new(&module.name, &ctx);
        let builder = Builder::new(&ctx);
        for func in module.fns.values() {
            _module.add_function(func.ty.to_llvm_func_type(&ctx), &func.name);
        }
        CodeGen {
            ctx,
            module: _module,
            builder,
        }
    }
    pub fn gen(module: &Module) {}
}
