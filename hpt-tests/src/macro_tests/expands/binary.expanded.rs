#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let c = a + b;
        Ok(c)
    }
    fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
        let c = &a + &b;
        Ok(c)
    }
    fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let c = a + __method_call_0;
        Ok(c)
    }
    fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let c = &a + &__method_call_0;
        Ok(c)
    }
    fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let __out_2 = &__method_call_0 / a;
        let c = &a + __out_2;
        Ok(c)
    }
    fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let __out_1 = &a / b;
        let __out_2 = &__method_call_0 / a;
        let c = __out_1 + __out_2;
        Ok(c)
    }
    fn case7(a: f32, b: f32) -> anyhow::Result<f32> {
        let __call_0 = compute(&a);
        let __try_2 = b.sin()?;
        let __try_4 = compute2(&__try_2)?;
        let __out_1 = __call_0 / b;
        let __out_2 = __try_4 / a;
        let c = __out_1 + __out_2;
        Ok(c)
    }
}
