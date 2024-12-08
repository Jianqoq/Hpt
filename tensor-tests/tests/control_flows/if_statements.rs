#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = if a > 0.0 {
                a + b
            } else {
                a - b
            };
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case2(a: f32, b: f32) -> anyhow::Result<f32>{
            Ok(if a > 0.0 {
                a + b
            } else {
                a - b
            })
        }
    );
    fuse_proc_macro!(
        fn case3(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = if a > 0.0 {
                let d = a + b;
                d
            } else {
                let d = a - b;
                d
            };
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case4(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = if a > 0.0 {
                let d = a + b;
                d + b.sin()
            } else {
                let d = a - b;
                d + b.sin()
            };
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case5(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = if a > 0.0 {
                let d = a + b;
                if d > 0.0 {
                    d + b.sin()
                } else {
                    d - b.sin()
                }
            } else {
                let d = a - b;
                if d > 0.0 {
                    d + b.sin()
                } else {
                    d - b.sin()
                }
            };
            Ok(c)
        }
    );
    fuse_proc_macro!(
        fn case6(a: f32, b: f32) -> anyhow::Result<f32>{
            let c = if a > 0.0 {
                let mut d = a + b;
                if d > 0.0 {
                    d = d.sin().sin()?.tanh()
                } else {
                    d = d.sin().sin()?.tanh()
                }
                d
            } else {
                let mut d = a + b;
                if d > 0.0 {
                    d = d.cos().cos()?.tanh()
                } else {
                    d = d.cos().cos()?.tanh()
                }
                d
            };
            Ok(c)
        }
    );
}
