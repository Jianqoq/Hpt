#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __for_out_0 = for i in 0..1000 {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __for_out_0 = for _ in (0..1000).iter() {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __for_out_0 = for _ in b.iter().enumerate() {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __for_out_0 = for _ in b.iter().enumerate() {
            a += 10;
            continue;
            break;
        };
        Ok(a)
    }
}
