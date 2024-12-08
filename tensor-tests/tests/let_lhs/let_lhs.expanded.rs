#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let Tensor { data: ok } = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let Tensor { data: ok, shape: _, strides: _, offset: _ } = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let Tensor { data: (ok, lk), shape: (_, _) } = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let Tensor {
            data: (ok, lk),
            shape: (Tensor { data: (ok, lk), shape: (_, _) }, _),
        } = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let (ok, lk) = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let (ok) = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let [ok, ok2] = a;
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let [ok, ok2, ..] = a;
        let __call_0 = Ok(a);
        __call_0
    }
}
