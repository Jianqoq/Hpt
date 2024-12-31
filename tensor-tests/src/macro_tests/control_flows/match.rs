#[macro_use]
extern crate tensor_tests;
use tensor_codegen::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            Ok(match a {
                10 => 10,
                20 => 20,
                30 => 30,
                _ => 40,
            })
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            Ok(match a {
                [1, 2, ..] => 10,
                _ => 20,
            })
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match a {
                Some(res) => res,
                None => 0.0,
            };
            Ok(res)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match a {
                Some(res) => {
                    if res > 0.0 {
                        10
                    } else {
                        20
                    }
                },
                None => 0.0,
            };
            Ok(res)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match a {
                Some(res) => if res > 0.0 {
                        10
                    } else {
                        20
                    },
                None => 0.0,
            };
            Ok(res)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match (a, b) {
                (Some(res), Some(res)) => if res > 0.0 {
                        10
                    } else {
                        20
                    },
                (_, _) => 0.0,
            };
            Ok(res)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match a {
                test::Tensor {filed, ..} => if filed > 0.0 {
                        10
                    } else {
                        20
                    },
                _ => 0.0,
            };
            Ok(res)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let res = match a {
                test::Tensor {filed, ..} => match b {
                    test::Tensor {filed2, ..} => if filed2 > 0.0 {
                        10
                    } else {
                        20
                    },
                    _ => 0.0,
                },
                _ => 0.0,
            };
            Ok(res)
        }
    );
}
