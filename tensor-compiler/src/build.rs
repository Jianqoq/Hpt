use crate::{halide::{code_gen::code_gen::CodeGen, executable::empty_executable::EmptyExecutable, module::Module}, opt_lvl::OptLvl, te::schedule::Schedule};


 
pub fn build(name: &str, schedules: &[Schedule], opt_lvl: OptLvl) -> EmptyExecutable {
    let mut module = Module::new(name);
    for i in schedules {
        module.add_function2(i);
    }
    let context = tensor_llvm::context::context::Context::new();
    let mut codegen = CodeGen::new(context, &module, opt_lvl as u32);
    codegen.compile();
    EmptyExecutable::new(codegen)
}