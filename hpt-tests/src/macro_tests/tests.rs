// #![allow(unused)]
// use hpt_codegen::fuse_proc_macro;

// #[test]
// fn test_binary_expands() {
//     fuse_proc_macro!(
//         fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = a + b;
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = &a + &b;
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = a + b.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = &a + &b.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = &a + &b.sin() / a;
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = &a / b + &b.sin() / a;
//             Ok(c)
//         }
//     );
//     fn compute(a: &f32) -> f32 {
//         *a
//     }
//     fn compute2(a: &f32) -> anyhow::Result<f32> {
//         Ok(*a)
//     }
//     fuse_proc_macro!(
//         fn case7(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = compute(&a) / b + compute2(&b.sin())? / a;
//             Ok(c)
//         }
//     );
//     macrotest::expand("src/macro_tests/expands/binary.rs");
// }

// #[test]
// fn test_expr_method_expands() {
//     fuse_proc_macro!(
//         fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = { a + b }.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = { &a + &b }.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = a + (b).sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = { &a + &b.sin() }.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = { &a + &b.sin() / a }.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = { &a / b + &b.sin() / a }.sin();
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case7(a: f32) -> anyhow::Result<f32> {
//             let c = a.sin().cos().tan();
//             Ok(c)
//         }
//     );
//     trait TestMethodCall {
//         fn op1(&self) -> anyhow::Result<f32>;
//         fn op2(&self, a: f32) -> anyhow::Result<f32>;
//         fn op3(&self, a: &f32) -> anyhow::Result<f32>;
//     }
//     impl TestMethodCall for f32 {
//         fn op1(&self) -> anyhow::Result<f32> {
//             unimplemented!()
//         }
//         fn op2(&self, _: f32) -> anyhow::Result<f32> {
//             unimplemented!()
//         }
//         fn op3(&self, _: &f32) -> anyhow::Result<f32> {
//             unimplemented!()
//         }
//     }
//     fuse_proc_macro!(
//         fn case8(a: f32) -> anyhow::Result<f32> {
//             let c = a.op1()?.op1()?.op1()?;
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case9(a: f32) -> anyhow::Result<f32> {
//             let c = a.op3(&a)?.op2(a)?.op2(a.op1()?)?;
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case10(a: f32) -> anyhow::Result<f32> {
//             let c = (0.0, {
//                 if a > 0.0 {
//                     a.op1()?
//                 } else {
//                     a.op1()?
//                 }
//             })
//                 .0
//                 .op3(&a)?
//                 .op2(a)?
//                 .op2(a.op1()?)?;
//             Ok(c)
//         }
//     );
//     macrotest::expand("src/macro_tests/expands/expr_method_call.rs");
// }

// #[test]
// fn test_if_statements() {
//     fuse_proc_macro!(
//         fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = if a > 0.0 { a + b } else { a - b };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
//             Ok(if a > 0.0 { a + b } else { a - b })
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(a: f32, b: f32) -> anyhow::Result<()> {
//             let c = if a > 0.0 {
//                 let d = a + b;
//             };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = if a > 0.0 {
//                 let d = a + b;
//                 d + b.sin()
//             } else {
//                 let d = a - b;
//                 d + b.sin()
//             };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = if a > 0.0 {
//                 let d = a + b;
//                 if d > 0.0 {
//                     d + b.sin()
//                 } else {
//                     d - b.sin()
//                 }
//             } else {
//                 let d = a - b;
//                 if d > 0.0 {
//                     d + b.sin()
//                 } else {
//                     d - b.sin()
//                 }
//             };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = if a > 0.0 {
//                 let mut d = a + b;
//                 if d > 0.0 {
//                     d = d.sin().sin().tanh()
//                 } else {
//                     d = d.sin().sin().tanh()
//                 }
//                 d
//             } else {
//                 let mut d = a + b;
//                 if d > 0.0 {
//                     d = d.cos().cos().tanh()
//                 } else {
//                     d = d.cos().cos().tanh()
//                 }
//                 d
//             };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case7(a: f32, b: f32) -> anyhow::Result<f32> {
//             let c = if a.sin() > 0.0 && a.cos() > 0.0 && {
//                 if b > 0.0 {
//                     true
//                 } else {
//                     false
//                 }
//             } {
//                 10.0
//             } else {
//                 20.0
//             };
//             Ok(c)
//         }
//     );
//     fuse_proc_macro!(
//         fn case8(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             if a > 0.0 {
//                 a += 10.0;
//             } else if a > 0.0 {
//                 a += 20.0;
//             } else {
//                 a += 30.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case9(a: f32, b: f32) -> anyhow::Result<f32> {
//             if a > 0.0 {
//                 10.0
//             } else if a > 0.0 {
//                 20.0
//             } else if a == 0.0 {
//                 30.0
//             } else {
//                 40.0
//             }
//             Ok(a)
//         }
//     );
//     macrotest::expand("src/macro_tests/control_flows/if_statements.rs");
// }

// #[test]
// fn test_for_loop() {
//     fuse_proc_macro!(
//         fn case1(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             for i in 0..1000 {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             for _ in (0..1000).collect::<Vec<_>>().iter() {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             for _ in (0..1000).collect::<Vec<_>>().iter().enumerate() {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             for _ in (0..1000).collect::<Vec<_>>().iter().enumerate() {
//                 a += 10.0;
//                 continue;
//                 break;
//             }
//             Ok(a)
//         }
//     );
//     macrotest::expand("src/macro_tests/control_flows/for_loop.rs");
// }

// #[test]
// fn test_match() {
//     fuse_proc_macro!(
//         fn case1(a: f32, b: f32) -> i32 {
//             match a {
//                 10.0 => 10,
//                 20.0 => 20,
//                 30.0 => 30,
//                 _ => 40,
//             }
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(a: [f32; 3], b: f32) -> i32 {
//             match a {
//                 [1.0, 2.0, ..] => 10,
//                 _ => 20,
//             }
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(a: Option<f32>, b: f32) -> f32 {
//             let res = match a {
//                 Some(res) => res,
//                 None => 0.0,
//             };
//             res
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(a: Option<f32>, b: f32) -> f32 {
//             let res = match a {
//                 Some(res) => {
//                     if res > 0.0 {
//                         10.0
//                     } else {
//                         20.0
//                     }
//                 }
//                 None => 0.0,
//             };
//             res
//         }
//     );
//     fuse_proc_macro!(
//         fn case5(a: Option<f32>, b: Option<f32>) -> f32 {
//             let res = match (a, b) {
//                 (Some(res), Some(res2)) => {
//                     if res > 0.0 {
//                         10.0
//                     } else {
//                         20.0
//                     }
//                 }
//                 (_, _) => 0.0,
//             };
//             res
//         }
//     );
//     macrotest::expand("src/macro_tests/control_flows/match.rs");
// }

// #[test]
// fn test_control_flow_mix() {
//     fuse_proc_macro!(
//         fn case1(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             while a < 1000.0 && a > 0.0 {
//                 a += 10.0;
//                 for i in (0..10).enumerate() {
//                     a += 10.0;
//                     if i.0 > 5 {
//                         break;
//                     } else if i.0 < 3 {
//                         continue;
//                     } else {
//                         a += 100.0;
//                     }
//                 }
//             }
//             Ok(a)
//         }
//     );
//     macrotest::expand("src/macro_tests/control_flows/mix.rs");
// }

// #[test]
// fn test_while_loop() {
//     fuse_proc_macro!(
//         fn case1(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             while a < 1000.0 && a > 0.0 {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case2(mut a: f32, b: f32) -> anyhow::Result<f32> {
//             let mut iter = (0..1000).collect::<Vec<_>>().iter();
//             while let Some(i) = iter.next() {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(mut a: f32, b: Vec<f32>) -> anyhow::Result<f32> {
//             while let Some(path) = b.iter().next() {
//                 a += 10.0;
//             }
//             Ok(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn case4(mut a: f32, b: Vec<f32>) -> anyhow::Result<f32> {
//             while let Some(path) = b.iter().next() {
//                 a += 10.0;
//                 continue;
//                 break;
//             }
//             Ok(a)
//         }
//     );
//     macrotest::expand("src/macro_tests/control_flows/while_loop.rs");
// }

// #[test]
// fn test_let_lhs() {
//     struct Tensor {
//         data: f32,
//     };
//     fuse_proc_macro!(
//         fn case1(a: Tensor) -> anyhow::Result<f32> {
//             let Tensor { data: b } = a;
//             Ok(b)
//         }
//     );
//     struct Tensor2 {
//         data: f32,
//         shape: (f32, f32),
//         strides: (f32, f32),
//         offset: f32,
//     };
//     fuse_proc_macro!(
//         fn case2(a: Tensor2) -> anyhow::Result<f32> {
//             let Tensor2 {
//                 data: b,
//                 shape: _,
//                 strides: _,
//                 offset: _,
//             } = a;
//             Ok(b)
//         }
//     );
//     fuse_proc_macro!(
//         fn case3(a: Tensor2) -> (f32, f32) {
//             let Tensor2 {
//                 data: _,
//                 shape: (b1, b2),
//                 strides: _,
//                 offset: _,
//             } = a;
//             (b1, b2)
//         }
//     );
//     struct Tensor4 {
//         data: (f32, f32),
//         shape: (Tensor2, Tensor2),
//     };
//     fuse_proc_macro!(
//         fn case4(a: Tensor4) -> (f32, f32, f32, f32) {
//             let Tensor4 {
//                 data: (b1, b2),
//                 shape:
//                     (
//                         Tensor2 {
//                             data: _,
//                             shape: (b3, b4),
//                             strides: _,
//                             offset: _,
//                         },
//                         _,
//                     ),
//             } = a;
//             (b1, b2, b3, b4)
//         }
//     );
//     fuse_proc_macro!(
//         fn case5(a: (f32, f32)) -> (f32, f32) {
//             let (ok, lk) = a;
//             (ok, lk)
//         }
//     );
//     fuse_proc_macro!(
//         fn case6(a: [f32; 2]) -> (f32, f32) {
//             let [ok, ok2] = a;
//             (ok, ok2)
//         }
//     );
//     fuse_proc_macro!(
//         fn case7(a: [f32; 3]) -> (f32, f32, f32) {
//             let [ok, ok2, ..] = a;
//             (ok, ok2, ok)
//         }
//     );
//     macrotest::expand("src/macro_tests/let_expr/let_lhs.rs");
// }

// #[test]
// fn test_let_rhs() {
//     fuse_proc_macro!(
//         fn array() {
//             let a = [1, 2, 3, 4];
//         }
//     );
//     fn compute() -> f32 {
//         1.0
//     }
//     fuse_proc_macro!(
//         fn assign() {
//             let a = compute();
//         }
//     );
//     fuse_proc_macro!(
//         fn async1() {
//             let a = async {};
//         }
//     );
//     use std::future::Future;
//     fuse_proc_macro!(
//         async fn await1(fut: impl Future<Output = f32>) {
//             let a = fut.await;
//         }
//     );
//     fuse_proc_macro!(
//         fn binary(a: i32, b: i32) {
//             let a = a + b;
//             let a = a - b;
//             let a = a * b;
//             let a = a / b;
//             let a = a % b;
//             let a = a & b;
//             let a = a | b;
//             let a = a ^ b;
//         }
//     );
//     fuse_proc_macro!(
//         fn unary(a: i32) {
//             let a = !a;
//             let a = -a;
//             let a = *&a;
//         }
//     );
//     fuse_proc_macro!(
//         fn block() {
//             let a = {
//                 let b = 1;
//                 b
//             };
//         }
//     );
//     fn invoke(a: i32, b: i32) -> i32 {
//         a + b
//     }
//     fuse_proc_macro!(
//         fn call(a: i32, b: i32) {
//             let a = invoke(a, b);
//         }
//     );
//     fuse_proc_macro!(
//         fn cast(a: f32) {
//             let a = a as f64;
//         }
//     );
//     fuse_proc_macro!(
//         fn closure(a: f32, b: f32) {
//             let a = |a: i32, b: i32| a + b;
//             let a = |a: i32, b: i32| a + b;
//         }
//     );
//     fuse_proc_macro!(
//         fn const1() {
//             let a = const {
//                 let b = 1;
//                 b
//             };
//         }
//     );
//     struct Obj {
//         k: f32,
//     }
//     struct Obj2(f32);
//     fuse_proc_macro!(
//         fn field(obj: Obj, obj2: Obj2) {
//             let a = obj.k;
//             let a = &obj.k;
//             let a = obj2.0;
//             let a = &obj2.0;
//         }
//     );
//     fuse_proc_macro!(
//         fn for_loop(expr: Vec<f32>) {
//             let a = for pat in expr {
//                 let b = 1;
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn if_expr(expr: bool) {
//             let a = if expr {
//                 let b = 1;
//             } else {
//                 let b = 2;
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn index(vec: Vec<f32>) {
//             let a = vec[2];
//         }
//     );
//     fuse_proc_macro!(
//         fn infer() {
//             let a = (0..100).collect::<Vec<_>>();
//         }
//     );
//     fuse_proc_macro!(
//         fn let_expr(opt: Option<f32>) {
//             if let Some(x) = opt {
//             } else {
//             }
//         }
//     );
//     fuse_proc_macro!(
//         fn lit() {
//             let a = 1;
//         }
//     );
//     fuse_proc_macro!(
//         fn loop1() {
//             let a = loop {
//                 let b = 1;
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn marcro1(q: f32) {
//             let a = format!("{}", q);
//         }
//     );
//     trait Foo {
//         fn foo<T>(&self, a: T, b: T) -> T;
//     }
//     impl Foo for f32 {
//         fn foo<T>(&self, a: T, b: T) -> T {
//             a
//         }
//     }
//     fuse_proc_macro!(
//         fn method_call(x: f32, a: f32, b: f32) {
//             let a = x.foo::<f32>(a, b);
//         }
//     );
//     fuse_proc_macro!(
//         fn match1(expr: f32) {
//             let a = match expr {
//                 _ => {
//                     let b = 1;
//                 }
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn range() {
//             let a = 1..2;
//         }
//     );
//     fuse_proc_macro!(
//         fn repeat() {
//             let a = [0; 10];
//         }
//     );
//     fuse_proc_macro!(
//         fn return1(a: f32) -> f32 {
//             return a;
//         }
//     );
//     struct Point {
//         x: i32,
//         y: i32,
//     }
//     struct Point2 {
//         x: Point,
//         y: i32,
//     }
//     fuse_proc_macro!(
//         fn struct1(a: i32, b: i32) {
//             let a0 = Point { x: 1, y: 1 };
//             let a1 = Point { x: a - b, y: 1 };
//             let a2 = Point2 {
//                 x: Point { x: a - b, y: 1 },
//                 y: 1,
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn try1(expr: Option<f32>) -> Option<f32> {
//             let a = expr?;
//             Some(a)
//         }
//     );
//     fuse_proc_macro!(
//         fn tuple1() {
//             let a = (1, 2, 3, 4);
//         }
//     );
//     fuse_proc_macro!(
//         fn unsafe1() {
//             let a = unsafe {
//                 let b = 1;
//             };
//         }
//     );
//     fuse_proc_macro!(
//         fn while1(expr: bool) {
//             let a = while expr {
//                 let b = 1;
//             };
//         }
//     );
//     macrotest::expand("src/macro_tests/let_expr/let_rhs.rs");
// }

// #[test]
// fn test_macros() {
//     macrotest::expand("src/macro_tests/macros/*.rs");
// }
