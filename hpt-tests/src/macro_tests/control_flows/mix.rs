#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            while a < 1000 && a > 0 {
                a += 10;
                for i in (0..10).iter().enumerate() {
                    a += 10;
                    if i.0 > 5 {
                        break;
                    } else if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    }
                }
            }
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            while let Some(i) = (0..1000).iter().next() {
                a += 10;
                for i in (0..10).iter().enumerate() {
                    a += 10;
                    if let Some(x) = i.iter().next() {
                        break;
                    } else if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    }
                }
            }
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            while let syn::Expr::Path(path) = b.iter().next() {
                a += 10;
                for i in ({a + b}..{if a > 0 {100} else {200}}).iter().enumerate() {
                    a += 10;
                    if let Some(x) = i.iter().next() {
                        break;
                    } else if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    }
                }
            }
            Ok(a)
        }
    );
}
