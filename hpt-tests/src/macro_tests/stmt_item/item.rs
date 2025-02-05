#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn struct_item(a: f32, b: f32) -> anyhow::Result<f32>{
            struct A {
                a: f32,
                b: f32,
            }
            Ok(A { a, b })
        }
    );
    fuse_proc_macro!(
        fn macro_item(a: f32, b: f32) -> anyhow::Result<f32>{
            macro_rules! a {
                ($a:expr) => {
                    $a
                };
            }
            Ok(a!(a))
        }
    );
    fuse_proc_macro!(
        fn trait_item(a: f32, b: f32) -> anyhow::Result<f32>{
            trait A {
                fn a(&self) -> f32;
            }
            Ok(a)
        }
    );
}
