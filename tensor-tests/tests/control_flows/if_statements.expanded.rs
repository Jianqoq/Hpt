#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 { a + b } else { a - b };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case2(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 { a + b } else { a - b };
        Ok(__if_assign_0)
    }
    fn case3(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            let d = a + b;
        };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case4(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            let d = a + b;
            let __method_call_1 = b.sin();
            d + __method_call_1
        } else {
            let d = a - b;
            let __method_call_2 = b.sin();
            d + __method_call_2
        };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case5(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            let d = a + b;
            let __if_assign_1 = if d > 0.0 {
                let __method_call_2 = b.sin();
                d + __method_call_2
            } else {
                let __method_call_3 = b.sin();
                d - __method_call_3
            };
            __if_assign_1
        } else {
            let d = a - b;
            let __if_assign_4 = if d > 0.0 {
                let __method_call_5 = b.sin();
                d + __method_call_5
            } else {
                let __method_call_6 = b.sin();
                d - __method_call_6
            };
            __if_assign_4
        };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            let mut d = a + b;
            let __if_assign_1 = if d > 0.0 {
                let __method_call_2 = d.sin();
                let __try_4 = __method_call_2.sin()?;
                let __method_call_5 = __try_4.tanh();
                d = __method_call_5;
            } else {
                let __method_call_6 = d.sin();
                let __try_8 = __method_call_6.sin()?;
                let __method_call_9 = __try_8.tanh();
                d = __method_call_9;
            };
            d
        } else {
            let mut d = a + b;
            let __if_assign_10 = if d > 0.0 {
                let __method_call_11 = d.cos();
                let __try_13 = __method_call_11.cos()?;
                let __method_call_14 = __try_13.tanh();
                d = __method_call_14;
            } else {
                let __method_call_15 = d.cos();
                let __try_17 = __method_call_15.cos()?;
                let __method_call_18 = __try_17.tanh();
                d = __method_call_18;
            };
            d
        };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case6(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a.shape() > 0.0 && a.len() > 0
            && { if i > 0 { 0 } else { 1 } }
        {
            d
        } else {
            d
        };
        let c = __if_assign_0;
        Ok(c)
    }
    fn case7(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            a += 10;
        } else {
            let __if_assign_1 = if a > 0.0 {
                a += 20;
            } else {
                a += 30;
            };
            __if_assign_1
        };
        Ok(a)
    }
    fn case8(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if a > 0.0 {
            10
        } else {
            let __if_assign_1 = if a > 0.0 {
                20
            } else {
                let __if_assign_2 = if a == 0.0 { 30 } else { 40 };
                __if_assign_2
            };
            __if_assign_1
        };
        Ok(a)
    }
}
