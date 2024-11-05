use proc_macro::TokenStream;
use proc_macro2::Literal;
use quote::quote;
use regex::Regex;
use syn::{ parse::{ Parse, ParseStream }, parse_macro_input, Expr, Ident, Token };

pub(crate) fn conv2d_microkernel_declare_const(inputs: TokenStream) -> TokenStream {
    let fn_name = parse_macro_input!(inputs as Ident);
    let text = fn_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        // 获取 x 前后的两个数字
        let before_x = captures.get(1).unwrap().as_str();
        let after_x = captures.get(2).unwrap().as_str();
        let before_x = before_x.parse::<Literal>().unwrap();
        let after_x = after_x.parse::<Literal>().unwrap();
        return (
            quote! {
            const OW_BLOCK: usize = #before_x;
            const OC_BLOCK: usize = #after_x;
        }
        ).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<Literal>().unwrap();
            return (quote! {
                const OW_BLOCK: usize = #before_x;
            }).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

struct ParseInpArgs {
    name: Ident,
    is3: Ident,
    step_width_m: Expr,
    template_name: Ident,
}

impl Parse for ParseInpArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let is3 = input.parse()?;
        input.parse::<Token![,]>()?;
        let step_width_m = input.parse()?;
        input.parse::<Token![,]>()?;
        let template_name = input.parse()?;
        Ok(Self {
            name,
            is3,
            step_width_m,
            template_name,
        })
    }
}

struct ParseKernelArgs {
    name: Ident,
    template_name: Ident,
}

impl Parse for ParseKernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let template_name = input.parse()?;
        Ok(Self {
            name,
            template_name,
        })
    }
}

struct ParseResultsArgs {
    name: Ident,
    inp: Ident,
    kernel_vecs: Ident,
    template_name: Ident,
}

impl Parse for ParseResultsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let inp = input.parse()?;
        input.parse::<Token![,]>()?;
        let kernel_vecs = input.parse()?;
        input.parse::<Token![,]>()?;
        let template_name = input.parse()?;
        Ok(Self {
            name,
            inp,
            kernel_vecs,
            template_name,
        })
    }
}

pub(crate) fn conv2d_microkernel_gen_inps(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseInpArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<i64>().unwrap();
        let arr = (0..before_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        let is3 = inp_args.is3;
        let step_width_m = inp_args.step_width_m;
        return (quote! {
            repeat_inp!(#inp, #is3, #step_width_m, [#arr])
        }).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<i64>().unwrap();
            let arr = (0..before_x).map(|i| i);
            let arr =
                quote! {
                #(
                    #arr
                ),*
            };
            let inp = inp_args.name;
            let is3 = inp_args.is3;
            let step_width_m = inp_args.step_width_m;
            return (
                quote! {
                repeat_inp!(#inp, #is3, #step_width_m, [#arr])
            }
            ).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn conv2d_microkernel_gen_pad_inps(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseInpArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<i64>().unwrap();
        let arr = (0..before_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        let is3 = inp_args.is3;
        return (
            quote! {
            repeat_pad_inp!(
                #inp,
                #is3,
                k,
                step_width,
                m,
                dw,
                isw,
                img_width,
                pw_start,
                l_in_range,
                [#arr]
            )
        }
        ).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<i64>().unwrap();
            let arr = (0..before_x).map(|i| i);
            let arr =
                quote! {
                #(
                    #arr
                ),*
            };
            let inp = inp_args.name;
            let is3 = inp_args.is3;
            return (
                quote! {
                repeat_pad_inp!(
                    #inp,
                    #is3,
                    k,
                    step_width,
                    m,
                    dw,
                    isw,
                    img_width,
                    pw_start,
                    l_in_range,
                    [#arr]
                )
            }
            ).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn transpose_conv2d_microkernel_gen_pad_inps(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseInpArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<i64>().unwrap();
        let arr = (0..before_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        let is3 = inp_args.is3;
        return (
            quote! {
            repeat_pad_inp!(
                #inp,
                #is3,
                k,
                step_width,
                m,
                dw,
                isw,
                out_width,
                pw_start,
                l_in_range,
                [#arr]
            )
        }
        ).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<i64>().unwrap();
            let arr = (0..before_x).map(|i| i);
            let arr =
                quote! {
                #(
                    #arr
                ),*
            };
            let inp = inp_args.name;
            let is3 = inp_args.is3;
            return (
                quote! {
                repeat_pad_inp!(
                    #inp,
                    #is3,
                    k,
                    step_width,
                    m,
                    dw,
                    isw,
                    out_width,
                    pw_start,
                    l_in_range,
                    [#arr]
                )
            }
            ).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn dwconv2d_microkernel_gen_pad_inps(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseInpArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<i64>().unwrap();
        let arr = (0..before_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        let is3 = inp_args.is3;
        return (
            quote! {
            repeat_pad_inp!(
                #inp,
                #is3,
                k,
                step_width,
                m,
                dw,
                isw,
                img_width,
                pw_start,
                l_in_range,
                ic_remain,
                [#arr]
            )
        }
        ).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<i64>().unwrap();
            let arr = (0..before_x).map(|i| i);
            let arr =
                quote! {
                #(
                    #arr
                ),*
            };
            let inp = inp_args.name;
            let is3 = inp_args.is3;
            return (
                quote! {
                repeat_pad_inp!(
                    #inp,
                    #is3,
                    k,
                    step_width,
                    m,
                    dw,
                    isw,
                    img_width,
                    pw_start,
                    l_in_range,
                    ic_remain,
                    [#arr]
                )
            }
            ).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn pwconv2d_microkernel_gen_pad_inps(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseInpArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<i64>().unwrap();
        let arr = (0..before_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        let is3 = inp_args.is3;
        (
            quote! {
                repeat_pw_pad_inp!(
                #inp,
                #is3,
                k,
                step_width,
                isw,
                img_width,
                pw_start,
                l_in_range,
                [#arr]
            )
        }
        ).into()
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<i64>().unwrap();
            let arr = (0..before_x).map(|i| i);
            let arr =
                quote! {
                #(
                    #arr
                ),*
            };
            let inp = inp_args.name;
            let is3 = inp_args.is3;
            (
                quote! {
                    repeat_pw_pad_inp!(
                    #inp,
                    #is3,
                    k,
                    step_width,
                    isw,
                    img_width,
                    pw_start,
                    l_in_range,
                    [#arr]
                )
            }
            ).into()
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn conv2d_microkernel_gen_kernels(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseKernelArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let after_x = captures.get(2).unwrap().as_str();
        let after_x = after_x.parse::<usize>().unwrap();
        let arr = (0..after_x).map(|i| i);
        let arr = quote! {
            #(
                #arr
            ),*
        };
        let inp = inp_args.name;
        return (quote! {
            repeat_kernel!(#inp, [#arr])
        }).into();
    } else {
        panic!("Invalid input format, must contains format like 5x1 or 5_1");
    }
}

pub(crate) fn conv2d_microkernel_gen_results(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseResultsArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<usize>().unwrap();
        let after_x = captures.get(2).unwrap().as_str();
        let after_x = after_x.parse::<usize>().unwrap();
        let ow_arr = (0..before_x).map(|i| {
            let unsuffixed = Literal::usize_unsuffixed(i);
            quote! {
                #unsuffixed
            }
        });
        let ow_arr = quote! {
            #(
                #ow_arr
            ),*
        };
        let oc_arr = (0..after_x).map(|i| {
            let unsuffixed = Literal::usize_unsuffixed(i);
            quote! {
                #unsuffixed
            }
        });
        let oc_arr = quote! {
            #(
                #oc_arr
            ),*
        };
        let results = inp_args.name;
        let inp = inp_args.inp;
        let kernel_vecs = inp_args.kernel_vecs;
        return (
            quote! {
            repeat_results!(#results, #inp, #kernel_vecs, [#oc_arr], [#ow_arr])
        }
        ).into();
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<usize>().unwrap();
            let ow_arr = (0..before_x).map(|i| {
                let unsuffixed = Literal::usize_unsuffixed(i);
                quote! {
                    #unsuffixed
                }
            });
            let ow_arr =
                quote! {
                #(
                    #ow_arr
                ),*
            };
            let results = inp_args.name;
            let inp = inp_args.inp;
            let kernel_vecs = inp_args.kernel_vecs;
            return (
                quote! {
                repeat_results!(#results, #inp, #kernel_vecs, [0], [#ow_arr])
            }
            ).into();
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}

pub(crate) fn dwconv2d_microkernel_gen_results(inputs: TokenStream) -> TokenStream {
    let inp_args = parse_macro_input!(inputs as ParseResultsArgs);
    let text = inp_args.template_name.to_string();
    let re = Regex::new(r"(\d+)x(\d+)").unwrap();
    if let Some(captures) = re.captures(&text) {
        let before_x = captures.get(1).unwrap().as_str();
        let before_x = before_x.parse::<usize>().unwrap();
        let ow_arr = (0..before_x).map(|i| {
            let unsuffixed = Literal::usize_unsuffixed(i);
            quote! {
                #unsuffixed
            }
        });
        let ow_arr = quote! {
            #(
                #ow_arr
            ),*
        };
        let results = inp_args.name;
        let inp = inp_args.inp;
        let kernel_vecs = inp_args.kernel_vecs;
        (
            quote! {
            repeat_results!(#results, #inp, #kernel_vecs, i, [#ow_arr])
        }
        ).into()
    } else {
        let re = Regex::new(r"(\d+)_(\d+)").unwrap();
        if let Some(captures) = re.captures(&text) {
            let before_x = captures.get(1).unwrap().as_str();
            let before_x = before_x.parse::<usize>().unwrap();
            let ow_arr = (0..before_x).map(|i| {
                let unsuffixed = Literal::usize_unsuffixed(i);
                quote! {
                    #unsuffixed
                }
            });
            let ow_arr =
                quote! {
                #(
                    #ow_arr
                ),*
            };
            let results = inp_args.name;
            let inp = inp_args.inp;
            let kernel_vecs = inp_args.kernel_vecs;
            (
                quote! {
                repeat_results!(#results, #inp, #kernel_vecs, i, [#ow_arr])
            }
            ).into()
        } else {
            panic!("Invalid input format, must contains format like 5x1 or 5_1");
        }
    }
}
