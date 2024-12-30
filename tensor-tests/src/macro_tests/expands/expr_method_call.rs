#[macro_use]
extern crate tensor_tests;
use tensor_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = {a + b}.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case2(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = {&a + &b}.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case3(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = a + (b).sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case4(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = {&a + &b.sin()}.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case5(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = {&a + &b.sin() / a}.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case6(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = {&a / b + &b.sin() / a}.sin();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case7(a: f32) -> anyhow::Result<f32>{
            let c = a.sin().cos().tan();
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case8(a: f32) -> anyhow::Result<f32>{
            let c = a.sin()?.cos()?.tan()?;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case9(a: f32) -> anyhow::Result<f32>{
            let c = a.sin(&a)?.cos(a)?.tan(a.selu()?)?;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case10(a: f32) -> anyhow::Result<f32>{
            let c = (
                0, {
                    if a > 0.0 {
                        a.sin()?
                    } else {
                        a.cos()?
                    }
                }
            ).sin(&a)?.cos(a)?.tan(a.selu()?)?;
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case10(a: f32) -> anyhow::Result<f32>{
            let c = (
                0, {
                    if a > 0.0 {
                        a.sin()?;
                    } else {
                        a.cos()?;
                    }
                }
            ).sin(&a)?.cos(a)?.tan(a.selu()?)?;
            Ok(c)
        }
    );
}
