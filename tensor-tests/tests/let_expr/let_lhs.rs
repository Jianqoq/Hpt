#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let Tensor {
                data: ok
            } = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let Tensor {
                data: ok,
                shape: _,
                strides: _,
                offset: _,
            } = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let Tensor {
                data: (ok, lk),
                shape: (_, _),
            } = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let Tensor {
                data: (ok, lk),
                shape: (Tensor {
                    data: (ok, lk),
                    shape: (_, _),
                }, _),
            } = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let (ok, lk) = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let (ok) = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let [ok, ok2] = a;
            Ok(a)
        }
    );
    fuse_proc_macro!(
        fn case1(a: f32, b: f32) -> anyhow::Result<f32>{
            let [ok, ok2, ..] = a;
            Ok(a)
        }
    );
}
