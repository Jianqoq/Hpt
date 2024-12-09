#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = { a + b };
        let c = __block_out_0.sin();
        let __call_1 = Ok(c);
        __call_1
    }
    fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = { &a + &b };
        let c = __block_out_0.sin();
        let __call_1 = Ok(c);
        __call_1
    }
    fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let c = a + __method_call_0;
        let __call_1 = Ok(c);
        __call_1
    }
    fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            &a + &__method_call_1
        };
        let c = __block_out_0.sin();
        let __call_2 = Ok(c);
        __call_2
    }
    fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            let __out_2 = &__method_call_1 / a;
            &a + __out_2
        };
        let c = __block_out_0.sin();
        let __call_2 = Ok(c);
        __call_2
    }
    fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            let __out_1 = &a / b;
            let __out_2 = &__method_call_1 / a;
            __out_1 + __out_2
        };
        let c = __block_out_0.sin();
        let __call_2 = Ok(c);
        __call_2
    }
    fn case7(a: f32) -> anyhow::Result<f32> {
        let __method_call_0 = a.sin();
        let __method_call_1 = __method_call_0.cos();
        let c = __method_call_1.tan();
        let __call_2 = Ok(c);
        __call_2
    }
    fn case8(a: f32) -> anyhow::Result<f32> {
        let __try_1 = a.sin()?;
        let __try_3 = __try_1.cos()?;
        let c = __try_3.tan()?;
        let __call_5 = Ok(c);
        __call_5
    }
    fn case9(a: f32) -> anyhow::Result<f32> {
        let __try_1 = a.sin(&a)?;
        let __try_3 = __try_1.cos(a)?;
        let __try_5 = a.selu()?;
        let c = __try_3.tan(__try_5)?;
        let __call_7 = Ok(c);
        __call_7
    }
    fn case10(a: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __if_assign_1 = if a > 0.0 {
                let __try_3 = a.sin()?;
                __try_3
            } else {
                let __try_5 = a.cos()?;
                __try_5
            };
            __if_assign_1
        };
        let __expr_tuple_6 = (0, __block_out_0);
        let __try_8 = __expr_tuple_6.sin(&a)?;
        let __try_10 = __try_8.cos(a)?;
        let __try_12 = a.selu()?;
        let c = __try_10.tan(__try_12)?;
        let __call_14 = Ok(c);
        __call_14
    }
    fn case10(a: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __if_assign_1 = if a > 0.0 {
                let __try_3 = a.sin()?;
                __try_3;
            } else {
                let __try_5 = a.cos()?;
                __try_5;
            };
            __if_assign_1
        };
        let __expr_tuple_6 = (0, __block_out_0);
        let __try_8 = __expr_tuple_6.sin(&a)?;
        let __try_10 = __try_8.cos(a)?;
        let __try_12 = a.selu()?;
        let c = __try_10.tan(__try_12)?;
        let __call_14 = Ok(c);
        __call_14
    }
}
