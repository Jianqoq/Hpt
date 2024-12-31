#[macro_use]
extern crate tensor_tests;
use tensor_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn array(){
            let a = [1, 2, 3, 4];
        }
    );
    fuse_proc_macro!(
        fn assign(){
            let a = compute();
        }
    );
    fuse_proc_macro!(
        fn async1(){
            let a = async {};
        }
    );
    fuse_proc_macro!(
        fn await1(fut: impl Future<Output = f32>){
            let a = fut.await;
        }
    );
    fuse_proc_macro!(
        fn binary(a: f32, b: f32){
            let a = a + b;
            let a = a - b;
            let a = a * b;
            let a = a / b;
            let a = a % b;
            let a = a & b;
            let a = a | b;
            let a = a ^ b;
        }
    );
    fuse_proc_macro!(
        fn unary(a: f32){
            let a = !a;
            let a = -a;
            let a = *a;
        }
    );
    fuse_proc_macro!(
        fn block(){
            let a = {
                let b = 1;
                b
            };
        }
    );
    fuse_proc_macro!(
        fn break1(){
            let a = break;
        }
    );
    fuse_proc_macro!(
        fn call(a: f32, b: f32){
            let a = invoke(a, b);
        }
    );
    fuse_proc_macro!(
        fn cast(a: f32){
            let a = a as f64;
        }
    );
    fuse_proc_macro!(
        fn closure(a: f32, b: f32){
            let a = |a, b| a + b;
            let a = |a, b| {a + b};
        }
    );
    fuse_proc_macro!(
        fn const1(){
            let a = const {
                let b = 1;
                b
            };
        }
    );
    fuse_proc_macro!(
        fn continue1(){
            let a = continue;
        }
    );
    fuse_proc_macro!(
        fn field(obj: f32){
            let a = obj.k;
            let a = &obj.k;
            let a = obj.0;
            let a = &obj.0;
        }
    );
    fuse_proc_macro!(
        fn for_loop(expr: f32){
            let a = for pat in expr {
                let b = 1;
            };
        }
    );
    fuse_proc_macro!(
        fn if_expr(expr: f32){
            let a = if expr {
                let b = 1;
            } else {
                let b = 2;
            };
        }
    );
    fuse_proc_macro!(
        fn index(vec: Vec<f32>){
            let a = vec[2];
        }
    );
    fuse_proc_macro!(
        fn infer(){
            let a = _;
        }
    );
    fuse_proc_macro!(
        fn let_expr(opt: Option<f32>){
            let Some(x) = opt;
        }
    );
    fuse_proc_macro!(
        fn lit(){
            let a = 1;
        }
    );
    fuse_proc_macro!(
        fn loop1(){
            let a = loop {
                let b = 1;
            };
        }
    );
    fuse_proc_macro!(
        fn marcro1(q: f32){
            let a = format!("{}", q);
        }
    );
    fuse_proc_macro!(
        fn method_call(x: f32, a: f32, b: f32){
            let a = x.foo::<T>(a, b);
        }
    );
    fuse_proc_macro!(
        fn match1(expr: f32){
            let a = match expr {
                _ => {
                    let b = 1;
                }
            };
        }
    );
    fuse_proc_macro!(
        fn range(){
            let a = 1..2;
        }
    );
    fuse_proc_macro!(
        fn repeat(){
            let a = [0; 10];
        }
    );
    fuse_proc_macro!(
        fn return1(a: f32){
            return a;
        }
    );
    fuse_proc_macro!(
        fn struct1(a: f32, b: f32){
            let a = Point { x: 1, y: 1 };
            let a = Point { x: a - b, y: 1 };
            let a = Point { x: Point { x: a - b, y: 1 }, y: 1 };
        }
    );
    fuse_proc_macro!(
        fn try1(expr: f32){
            let a = expr?;
        }
    );
    fuse_proc_macro!(
        fn tuple1(){
            let a = (1, 2, 3, 4);
        }
    );
    fuse_proc_macro!(
        fn unsafe1(){
            let a = unsafe {
                let b = 1;
            };
        }
    );
    fuse_proc_macro!(
        fn while1(expr: f32){
            let a = while expr {
                let b = 1;
            };
        }
    );
    fuse_proc_macro!(
        fn yield1(expr: f32){
            let a = yield expr;
        }
    );
}

