#[macro_use]
extern crate tensor_tests;
use tensor_codegen::*;
pub fn main() {
    fn array() {
        let a = [1, 2, 3, 4];
    }
    fn assign() {
        let a = compute();
    }
    fn async1() {
        let __async_0 = async {};
        let a = __async_0;
    }
    fn await1(fut: impl Future<Output = f32>) {
        let a = fut.await;
    }
    fn binary(a: f32, b: f32) {
        let a = a + b;
        let a = a - b;
        let a = a * b;
        let a = a / b;
        let a = a % b;
        let a = a & b;
        let a = a | b;
        let a = a ^ b;
    }
    fn unary(a: f32) {
        let a = !a;
        let a = -a;
        let a = *a;
    }
    fn block() {
        let __block_out_0 = { 1 };
        let a = __block_out_0;
    }
    fn break1() {
        let a = break;
    }
    fn call(a: f32, b: f32) {
        let a = invoke(a, b);
    }
    fn cast(a: f32) {
        let a = a as f64;
    }
    fn closure(a: f32, b: f32) {
        let __closure_assign_0 = |a, b| { a + b };
        let a = __closure_assign_0;
        let __closure_assign_1 = |a, b| {
            let __block_out_2 = { a + b };
            __block_out_2
        };
        let a = __closure_assign_1;
    }
    fn const1() {
        let __const_0 = const { 1 };
        let a = __const_0;
    }
    fn continue1() {
        let a = continue;
    }
    fn field(obj: f32) {
        let a = obj.k;
        let a = &obj.k;
        let a = obj.0;
        let a = &obj.0;
    }
    fn for_loop(expr: f32) {
        let __for_out_0 = for pat in expr {
            let b = 1;
        };
        let a = __for_out_0;
    }
    fn if_expr(expr: f32) {
        let __if_assign_0 = if expr {
            let b = 1;
        } else {
            let b = 2;
        };
        let a = __if_assign_0;
    }
    fn index(vec: Vec<f32>) {
        let a = vec[2];
    }
    fn infer() {
        let a = _;
    }
    fn let_expr(opt: Option<f32>) {
        let Some(x) = opt;
    }
    fn lit() {
        let a = 1;
    }
    fn loop1() {
        let __loop_assign_0 = loop {
            let b = 1;
        };
        let a = __loop_assign_0;
    }
    fn marcro1(q: f32) {
        let a = ::alloc::__export::must_use({
            let res = ::alloc::fmt::format(format_args!("{0}", q));
            res
        });
    }
    fn method_call(x: f32, a: f32, b: f32) {
        let a = x.foo::<T>(a, b);
    }
    fn match1(expr: f32) {
        let __match_assign_0 = match expr {
            _ => {
                let __block_out_1 = {
                    let b = 1;
                };
                __block_out_1
            }
        };
        let a = __match_assign_0;
    }
    fn range() {
        let a = 1..2;
    }
    fn repeat() {
        let a = [0; 10];
    }
    fn return1(a: f32) {
        return a;
    }
    fn struct1(a: f32, b: f32) {
        let a = Point { x: 1, y: 1 };
        let a = Point { x: a - b, y: 1 };
        let __expr_struct_2 = Point { x: a - b, y: 1 };
        let a = Point { x: __expr_struct_2, y: 1 };
    }
    fn try1(expr: f32) {
        let a = expr?;
    }
    fn tuple1() {
        let a = (1, 2, 3, 4);
    }
    fn unsafe1() {
        let __unsafe_0 = unsafe {
            let b = 1;
        };
        let a = __unsafe_0;
    }
    fn while1(expr: f32) {
        let __while_out_0 = while expr {
            let b = 1;
        };
        let a = __while_out_0;
    }
    fn yield1(expr: f32) {
        let a = yield expr;
    }
}
