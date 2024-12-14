#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            for i in 0..1000 {
                a += 10;
            }
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            for _ in (0..1000).iter() {
                a += 10;
            }
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            for _ in b.iter().enumerate() {
                a += 10;
            }
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            for _ in b.iter().enumerate() {
                a += 10;
                continue;
                break;
            }
            Ok(a)
        }
    );
}
