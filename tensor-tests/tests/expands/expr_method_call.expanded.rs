#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = { a + b };
        let c = __block_out_0.sin();
        Ok(c)
    }
    fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = { &a + &b };
        let c = __block_out_0.sin();
        Ok(c)
    }
    fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
        let __method_call_0 = b.sin();
        let c = a + __method_call_0;
        Ok(c)
    }
    fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            &a + &__method_call_1
        };
        let c = __block_out_0.sin();
        Ok(c)
    }
    fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            let __out_2 = &__method_call_1 / a;
            &a + __out_2
        };
        let c = __block_out_0.sin();
        Ok(c)
    }
    fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
        let __block_out_0 = {
            let __method_call_1 = b.sin();
            let __out_1 = &a / b;
            let __out_2 = &__method_call_1 / a;
            __out_1 + __out_2
        };
        let c = __block_out_0.sin();
        Ok(c)
    }
    fn case7(a: f32) -> anyhow::Result<f32> {
        let __method_call_0 = a.sin();
        let __method_call_1 = __method_call_0.cos();
        let c = __method_call_1.tan();
        Ok(c)
    }
    fn case8(a: f32) -> anyhow::Result<f32> {
        let __try_1 = a.sin()?;
        let __try_3 = __try_1.cos()?;
        let c = __try_3.tan()?;
        Ok(c)
    }
    fn case9(a: f32) -> anyhow::Result<f32> {
        let __try_1 = a.sin(&a)?;
        let __try_3 = __try_1.cos(a)?;
        let __try_5 = a.selu()?;
        let c = __try_3.tan(__try_5)?;
        Ok(c)
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
        let __expr_tuple = (0, __block_out_0);
        let __try_7 = __expr_tuple.sin(&a)?;
        let __try_9 = __try_7.cos(a)?;
        let __try_11 = a.selu()?;
        let c = __try_9.tan(__try_11)?;
        Ok(c)
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
        let __expr_tuple = (0, __block_out_0);
        let __try_7 = __expr_tuple.sin(&a)?;
        let __try_9 = __try_7.cos(a)?;
        let __try_11 = a.selu()?;
        let c = __try_9.tan(__try_11)?;
        Ok(c)
    }
}
