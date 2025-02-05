#[macro_use]
extern crate hpt_tests;
use hpt_codegen::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while a < 1000 && a > 0 {
            a += 10;
            let __for_out_1 = for i in (0..10).iter().enumerate() {
                a += 10;
                let __if_assign_2 = if i.0 > 5 {
                    break;
                } else {
                    let __if_assign_3 = if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    };
                    __if_assign_3
                };
                __if_assign_2
            };
            __for_out_1
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while let Some(i) = (0..1000).iter().next() {
            a += 10;
            let __for_out_1 = for i in (0..10).iter().enumerate() {
                a += 10;
                let __if_assign_2 = if let Some(x) = i.iter().next() {
                    break;
                } else {
                    let __if_assign_3 = if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    };
                    __if_assign_3
                };
                __if_assign_2
            };
            __for_out_1
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while let syn::Expr::Path(path) = b.iter().next() {
            a += 10;
            let __for_out_1 = for i in ({ a + b }..{ if a > 0 { 100 } else { 200 } })
                .iter()
                .enumerate()
            {
                a += 10;
                let __if_assign_2 = if let Some(x) = i.iter().next() {
                    break;
                } else {
                    let __if_assign_3 = if i.0 < 3 {
                        continue;
                    } else {
                        a += 100;
                    };
                    __if_assign_3
                };
                __if_assign_2
            };
            __for_out_1
        };
        Ok(a)
    }
}
