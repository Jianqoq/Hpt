#[macro_use]
extern crate tensor_tests;
use tensor_dyn::*;
pub fn main() {
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        {
            ::std::io::_print(format_args!("case1\n"));
        };
        let __call_0 = Ok(a);
        __call_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __macro_0 = {
            ::std::io::_print(format_args!("case2\n"));
        };
        let __call_1 = Ok(__macro_0);
        __call_1
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __macro_0 = {
            ::std::io::_print(format_args!("case3\n"));
        };
        __macro_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __macro_0 = {
            ::std::io::_print(format_args!("case4\n"));
        };
        __macro_0 + 10
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __macro_0 = {
            ::std::io::_print(format_args!("case5\n"));
        };
        let __macro_1 = {
            ::std::io::_print(format_args!("case6\n"));
        };
        __macro_0 + __macro_1
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __if_assign_0 = if {
            ::std::io::_print(format_args!("case7\n"));
        } {
            let __macro_1 = {
                ::std::io::_print(format_args!("case8\n"));
            };
            __macro_1
        };
        __if_assign_0
    }
    fn case1(a: f32, b: f32) -> anyhow::Result<f32> {
        let __match_assign_0 = match {
            ::std::io::_print(format_args!("case9\n"));
        } {
            _ => {
                let __macro_1 = {
                    ::std::io::_print(format_args!("case10\n"));
                };
                __macro_1
            }
        };
        __match_assign_0
    }
}
