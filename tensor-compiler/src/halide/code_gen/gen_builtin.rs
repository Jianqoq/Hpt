use tensor_llvm::{
    context::context::Context,
    module::module::Module,
    types::values::FunctionValue,
};
use tensor_types::dtype::Dtype;

pub fn uary_gen(
    res_type: Dtype,
    ctx: &Context,
    module: &Module,
    fn_key: &'static str
) -> FunctionValue {
    match res_type {
        Dtype::F32 => {
            let key = format!("{}f", fn_key);
            let f32_type = ctx.f32_type();
            let fn_type = f32_type.fn_type(&[f32_type.into()], false);
            let function = module.add_function(fn_type, &key);
            function
        }
        Dtype::F64 => {
            let f64_type = ctx.f64_type();
            let fn_type = f64_type.fn_type(&[f64_type.into()], false);
            let function = module.add_function(fn_type, fn_key);
            function
        }
        _ => unimplemented!(),
    }
}
