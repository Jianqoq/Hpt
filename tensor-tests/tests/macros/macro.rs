#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            println!("case1");
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            Ok(println!("case2"))
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            println!("case3")
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            println!("case4") + 10
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            println!("case5") + println!("case6")
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            if println!("case7") {
                println!("case8")
            }
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            match println!("case9") {
                _ => println!("case10")
            }
        }
    );
}
