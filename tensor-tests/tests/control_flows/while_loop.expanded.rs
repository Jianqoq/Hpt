#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while a < 1000 && a > 0 {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while (let Some(i) = (0..1000).iter().next()) {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while (let syn::Expr::Path(path) = b.iter().next()) {
            a += 10;
        };
        Ok(a)
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __while_out_0 = while (let syn::Expr::Path(path) = b.iter().next()) {
            a += 10;
            continue;
            break;
        };
        Ok(a)
    }
}
