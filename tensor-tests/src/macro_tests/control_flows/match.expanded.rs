#[macro_use]
extern crate tensor_tests;
use tensor_codegen::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            10 => 10,
            20 => 20,
            30 => 30,
            _ => 40,
        };
        Ok(__match_assign_0)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            [1, 2, ..] => 10,
            _ => 20,
        };
        Ok(__match_assign_0)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            Some(res) => res,
            None => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            Some(res) => {
                let __block_out_1 = {
                    let __if_assign_2 = if res > 0.0 { 10 } else { 20 };
                    __if_assign_2
                };
                __block_out_1
            }
            None => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            Some(res) => {
                let __if_assign_1 = if res > 0.0 { 10 } else { 20 };
                __if_assign_1
            }
            None => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match (a, b) {
            (Some(res), Some(res)) => {
                let __if_assign_1 = if res > 0.0 { 10 } else { 20 };
                __if_assign_1
            }
            (_, _) => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            test::Tensor { filed, .. } => {
                let __if_assign_1 = if filed > 0.0 { 10 } else { 20 };
                __if_assign_1
            }
            _ => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match a {
            test::Tensor { filed, .. } => {
                let __match_assign_1 = match b {
                    test::Tensor { filed2, .. } => {
                        let __if_assign_2 = if filed2 > 0.0 { 10 } else { 20 };
                        __if_assign_2
                    }
                    _ => 0.0,
                };
                __match_assign_1
            }
            _ => 0.0,
        };
        let res = __match_assign_0;
        Ok(res)
    }
}
