use std::{ borrow::Cow, ffi::{ CStr, CString } };

use crate::{
    builder::builder::Builder,
    context::context::Context,
    module::module::Module,
    types::values::BasicValue,
};

pub fn to_c_str<'s>(mut s: &'s str) -> Cow<'s, CStr> {
    if s.is_empty() {
        s = "\0";
    }

    if
        !s
            .chars()
            .rev()
            .any(|ch| ch == '\0')
    {
        return Cow::from(CString::new(s).expect("unreachable since null bytes are checked"));
    }

    unsafe { Cow::from(CStr::from_ptr(s.as_ptr() as *const _)) }
}

pub fn declare_printf(context: &Context, module: &Module) {
    let i32_type = context.i32_type();
    let i8_type = context.i8_type();
    let fn_type = i32_type.fn_type(&[i8_type.into()], true);
    module.add_function(fn_type, "printf");
}

#[allow(dead_code)]
pub fn insert_printf(
    ctx: &Context,
    builder: &Builder,
    module: &Module,
    fmt: &str,
    args: &[BasicValue]
) {
    let global_str = builder.build_global_string_ptr(fmt, "fmt").to_ptr_value();
    let format_str_ptr = builder.build_gep(
        ctx.str_ptr_type(),
        global_str,
        &[ctx.i32_type().const_int(0, false).into(), ctx.i32_type().const_int(0, false).into()],
        "fmtstr"
    );
    let mut print_args = vec![format_str_ptr.into()];
    args.iter().for_each(|arg| print_args.push((*arg).into()));
    let printf = ctx.i32_type().fn_type(&[ctx.i8_type().into()], true);
    let func = module.get_function(printf, "printf").unwrap();
    builder.build_call(&func, &print_args, "printfCall");
}
