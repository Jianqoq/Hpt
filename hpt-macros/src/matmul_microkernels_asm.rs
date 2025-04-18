use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Token;
use syn::{
    parse::{Parse, ParseStream},
    Ident, LitInt,
};

pub(crate) struct MatmulMicrokernelArgs {
    pub(crate) mr: usize,
    pub(crate) nr: usize,
    pub(crate) vxor: String,
    pub(crate) vmov_unaligned: String,
    pub(crate) vmov_aligned: String,
    pub(crate) vbroadcast: String,
    pub(crate) vfma: String,
    pub(crate) vadd: String,
    pub(crate) a_ptr: Ident,
    pub(crate) b_ptr: Ident,
    pub(crate) c_ptr: Ident,
    pub(crate) lda: String,
    pub(crate) ldc: String,
    pub(crate) vec_size: usize,
    pub(crate) c_regs: Vec<Vec<Ident>>,
    pub(crate) b_regs: Vec<Ident>,
    pub(crate) a_regs: Vec<Ident>,
    pub(crate) res_regs: Vec<Ident>,
    pub(crate) kc: String,
    pub(crate) ks: String,
    pub(crate) first_kiter: String,
}

impl Parse for MatmulMicrokernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let nr_lit = input.parse::<LitInt>()?;
        let nr = nr_lit.base10_parse::<usize>()?;
        input.parse::<Token![,]>()?;

        let mr_lit = input.parse::<LitInt>()?;
        let mr = mr_lit.base10_parse::<usize>()?;
        input.parse::<Token![,]>()?;

        let vec_size_lit = input.parse::<LitInt>()?;
        let vec_size = vec_size_lit.base10_parse::<usize>()?;
        input.parse::<Token![,]>()?;

        let vxor_lit = input.parse::<Ident>()?;
        let vxor = vxor_lit.to_string();
        input.parse::<Token![,]>()?;

        let vmov_aligned_lit = input.parse::<Ident>()?;
        let vmov_aligned = vmov_aligned_lit.to_string();
        input.parse::<Token![,]>()?;

        let vmov_unaligned_lit = input.parse::<Ident>()?;
        let vmov_unaligned = vmov_unaligned_lit.to_string();
        input.parse::<Token![,]>()?;

        let vbroadcast_lit = input.parse::<Ident>()?;
        let vbroadcast = vbroadcast_lit.to_string();
        input.parse::<Token![,]>()?;

        let vfma_lit = input.parse::<Ident>()?;
        let vfma = vfma_lit.to_string();
        input.parse::<Token![,]>()?;

        let vadd_lit = input.parse::<Ident>()?;
        let vadd = vadd_lit.to_string();
        input.parse::<Token![,]>()?;

        let a_ptr = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;

        let b_ptr = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;

        let c_ptr = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;

        let lda_lit = input.parse::<Ident>()?;
        let lda = lda_lit.to_string();
        input.parse::<Token![,]>()?;

        let ldc_lit = input.parse::<Ident>()?;
        let ldc = ldc_lit.to_string();
        input.parse::<Token![,]>()?;

        let kc_lit = input.parse::<Ident>()?;
        let kc = kc_lit.to_string();
        input.parse::<Token![,]>()?;

        let ks_lit = input.parse::<Ident>()?;
        let ks = ks_lit.to_string();
        input.parse::<Token![,]>()?;

        let first_kiter_lit = input.parse::<Ident>()?;
        let first_kiter = first_kiter_lit.to_string();
        input.parse::<Token![,]>()?;

        let c_regs = input.parse::<syn::ExprArray>()?;
        input.parse::<Token![,]>()?;

        let b_regs = input.parse::<syn::ExprArray>()?;
        input.parse::<Token![,]>()?;

        let a_regs = input.parse::<syn::ExprArray>()?;
        input.parse::<Token![,]>()?;

        let res_regs = input.parse::<syn::ExprArray>()?;

        let c_regs = c_regs
            .elems
            .iter()
            .map(|e| {
                let expr_array = if let syn::Expr::Array(expr_array) = e {
                    expr_array
                } else {
                    panic!("expected c_regs array");
                };
                expr_array
                    .elems
                    .iter()
                    .map(|e| {
                        if let syn::Expr::Path(path) = e {
                            path.path.get_ident().expect("expected ident").clone()
                        } else {
                            panic!("expected ident");
                        }
                    })
                    .collect()
            })
            .collect();

        let b_regs = b_regs
            .elems
            .iter()
            .map(|e| {
                if let syn::Expr::Path(path) = e {
                    path.path.get_ident().expect("expected ident").clone()
                } else {
                    panic!("expected ident");
                }
            })
            .collect();

        let a_regs = a_regs
            .elems
            .iter()
            .map(|e| {
                if let syn::Expr::Path(path) = e {
                    path.path.get_ident().expect("expected ident").clone()
                } else {
                    panic!("expected ident");
                }
            })
            .collect();

        let res_regs = res_regs
            .elems
            .iter()
            .map(|e| {
                if let syn::Expr::Path(path) = e {
                    path.path.get_ident().expect("expected ident").clone()
                } else {
                    panic!("expected ident");
                }
            })
            .collect();

        Ok(MatmulMicrokernelArgs {
            mr,
            nr,
            vxor,
            vmov_unaligned,
            vmov_aligned,
            vbroadcast,
            vfma,
            vadd,
            a_ptr,
            b_ptr,
            c_ptr,
            lda,
            ldc,
            vec_size,
            c_regs,
            b_regs,
            a_regs,
            res_regs,
            kc,
            ks,
            first_kiter,
        })
    }
}

pub(crate) fn matmul_gen_asm(args: &MatmulMicrokernelArgs) -> TokenStream {
    let xor = &args.vxor;
    let vmov_unaligned = &args.vmov_unaligned;
    let vmov_aligned = &args.vmov_aligned;
    let vbroadcast = &args.vbroadcast;
    let vfma = &args.vfma;
    let a_ptr = args.a_ptr.to_string();
    let b_ptr = args.b_ptr.to_string();
    let c_ptr = args.c_ptr.to_string();
    let mr = args.mr;
    let nr = args.nr;
    let lda = args.lda.clone();
    let ldc = args.ldc.clone();
    let vec_size = args.vec_size;
    let a_regs = args.a_regs.clone();
    let kc = args.kc.clone();
    let first_kiter = args.first_kiter.clone();
    let res_regs = args.res_regs.clone();
    let vadd = args.vadd.clone();
    let ks = args.ks.clone();

    let mut asm = vec![];

    for regs in &args.c_regs {
        for reg in regs {
            asm.push(format!("{xor} {reg}, {reg}, {reg}"));
        }
    }

    asm.push(format!("test {{{kc}}}, {{{kc}}}"));
    asm.push(format!("jz 3f")); // if kc <= 0, jump to label 3

    asm.push(format!("2:"));

    for (idx, reg) in args.b_regs.iter().enumerate() {
        let offset = 4 * vec_size * idx;
        if offset == 0 {
            asm.push(format!("{vmov_aligned} {reg}, [{{{b_ptr}}}]"));
        } else {
            asm.push(format!("{vmov_aligned} {reg}, [{{{b_ptr}}} + {offset}]"));
        }
    }

    let mut reg_to_use = 0;
    for m in 0..mr {
        if m > 0 {
            asm.push(format!("add {{{a_ptr}}}, {{{lda}}}"));
        }
        let a_reg = &a_regs[reg_to_use];
        reg_to_use = (reg_to_use + 1) % a_regs.len();
        asm.push(format!("{vbroadcast} {a_reg}, [{{{a_ptr}}}]"));
        for n in 0..nr {
            let c_reg = &args.c_regs[m][n];
            let b_reg = &args.b_regs[n];
            let fma = format!("{vfma} {c_reg}, {a_reg}, {b_reg}");
            asm.push(fma);
        }
    }
    asm.push(format!(
        "add {{{b_ptr}}}, {}",
        4 * vec_size * nr
    ));
    asm.push(format!("add {{{a_ptr}}}, {{{ks}}}"));
    asm.push(format!("dec {{{kc}}}"));
    asm.push(format!("jnz 2b"));

    asm.push(format!("3:"));
    asm.push(format!("test {{{first_kiter}}}, {{{first_kiter}}}"));
    asm.push(format!("jz 4f"));

    for m in 0..mr {
        if m > 0 {
            asm.push(format!("add {{{c_ptr}}}, {{{ldc}}}"));
        }
        for n in 0..nr {
            let c_reg = &args.c_regs[m][n];
            let offset = n * vec_size * 4;
            let fma = if offset == 0 {
                format!("{vmov_unaligned} [{{{c_ptr}}}], {c_reg}")
            } else {
                format!("{vmov_unaligned} [{{{c_ptr}}} + {offset}], {c_reg}")
            };
            asm.push(fma);
        }
    }

    asm.push(format!("jmp 5f"));
    asm.push(format!("4:"));
    let mut reg_to_use = 0;
    for m in 0..mr {
        if m > 0 {
            asm.push(format!("add {{{c_ptr}}}, {{{ldc}}}"));
        }
        for n in 0..nr {
            let c_reg = &args.c_regs[m][n];
            let offset = n * vec_size * 4;
            let res_reg = &res_regs[reg_to_use];
            reg_to_use = (reg_to_use + 1) % res_regs.len();
            if offset == 0 {
                asm.push(format!("{vmov_unaligned} {res_reg}, [{{{c_ptr}}}]"));
                asm.push(format!("{vadd} {res_reg}, {res_reg}, {c_reg}"));
                asm.push(format!("{vmov_unaligned} [{{{c_ptr}}}], {res_reg}"));
            } else {
                asm.push(format!(
                    "{vmov_unaligned} {res_reg}, [{{{c_ptr}}} + {offset}]"
                ));
                asm.push(format!("{vadd} {res_reg}, {res_reg}, {c_reg}"));
                asm.push(format!(
                    "{vmov_unaligned} [{{{c_ptr}}} + {offset}], {res_reg}"
                ));
            }; // c_ptr[n * vec_size]
        }
    }

    asm.push("5:".to_string());

    let mut tokens = TokenStream::new();

    tokens.extend(asm.iter().map(|s| quote!(#s, )));

    let mut simd_regs = HashSet::new();
    for regs in &args.c_regs {
        for reg in regs {
            simd_regs.insert(reg.to_string());
        }
    }
    for reg in &args.b_regs {
        simd_regs.insert(reg.to_string());
    }
    for reg in &args.a_regs {
        simd_regs.insert(reg.to_string());
    }
    for reg in &args.res_regs {
        simd_regs.insert(reg.to_string());
    }

    let simd_regs_outs = simd_regs
        .iter()
        .map(|r| quote!(out(#r) _))
        .collect::<Vec<_>>();

    let lda_reg_ts = if mr > 1 {
        quote!(lda = in(reg) lda,)
    } else {
        quote!()
    };

    let ldc_reg_ts = if mr > 1 {
        quote!(ldc = in(reg) ldc,)
    } else {
        quote!()
    };

    quote! {
        std::arch::asm!(
            #tokens

            kc = in(reg) kc,
            b_ptr = in(reg) b_ptr,
            a_ptr = in(reg) a_ptr,
            c_ptr = in(reg) c_ptr,
            #lda_reg_ts
            #ldc_reg_ts
            first_kiter = in(reg) first_kiter,
            ks = in(reg) ks,
            #(#simd_regs_outs),*,
            options(nostack),
        )
    }
}
