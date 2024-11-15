use std::collections::HashMap;

use quote::ToTokens;
use syn::visit::Visit;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Type {
    Scalar,
    Tensor,
    Unknown,
}

#[derive(Debug, Clone)]
pub(crate) struct TyInfer {
    pub(crate) table: HashMap<String, Type>,
}

impl TyInfer {
    pub(crate) fn new() -> Self {
        Self { table: HashMap::new() }
    }
}

impl<'ast> Visit<'ast> for TyInfer {
    fn visit_signature(&mut self, i: &'ast syn::Signature) {
        for arg in i.inputs.iter() {
            if let syn::FnArg::Typed(pat_type) = arg {
                let pat = pat_type.pat.as_ref();
                let ty = pat_type.ty.to_token_stream().to_string();
                let tys = [
                    "bool",
                    "i8",
                    "i16",
                    "i32",
                    "i64",
                    "u8",
                    "u16",
                    "u32",
                    "u64",
                    "f32",
                    "f64",
                ];
                match pat {
                    syn::Pat::Ident(pat_ident) => {
                        let ident = pat_ident.ident.to_string();
                        if tys.iter().any(|t| ty.contains(t)) {
                            self.table.insert(ident, Type::Scalar);
                        } else if ident.contains("Tensor") || ident.contains("_Tensor") {
                            self.table.insert(ident, Type::Tensor);
                        } else {
                            self.table.insert(ident, Type::Unknown);
                        }
                    }
                    syn::Pat::Lit(_) => todo!(),
                    syn::Pat::Macro(_) => todo!(),
                    syn::Pat::Or(_) => todo!(),
                    syn::Pat::Paren(_) => todo!(),
                    syn::Pat::Path(_) => todo!(),
                    syn::Pat::Range(_) => todo!(),
                    syn::Pat::Reference(_) => todo!(),
                    syn::Pat::Rest(_) => todo!(),
                    syn::Pat::Slice(_) => todo!(),
                    syn::Pat::Struct(_) => todo!(),
                    syn::Pat::Tuple(_) => todo!(),
                    syn::Pat::TupleStruct(_) => todo!(),
                    syn::Pat::Type(_) => todo!(),
                    syn::Pat::Verbatim(_) => todo!(),
                    syn::Pat::Wild(_) => todo!(),
                    _ =>
                        unimplemented!(
                            "ty_infer::visit_signature::{}",
                            pat.to_token_stream().to_string()
                        ),
                }
            }
        }
    }
}
