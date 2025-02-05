#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = a + b;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case2(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = &a + &b;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case3(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = a + b.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case4(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = &a + &b.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case5(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = &a + &b.sin() / a;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case6(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = &a / b + &b.sin() / a;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case7(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = compute(&a) / b + compute2(&b.sin()?)? / a;
            Ok(c)
        }
    );
}