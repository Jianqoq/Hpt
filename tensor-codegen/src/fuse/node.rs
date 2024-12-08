use quote::ToTokens;

use super::kernel_type::KernelType;

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Unary {
    pub(crate) method: syn::Ident,
    pub(crate) operand: syn::Ident,
    pub(crate) args: Vec<syn::Expr>,
    pub(crate) output: syn::Ident,
    pub(crate) kernel_type: KernelType,
}

impl std::fmt::Debug for Unary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} = {}({}, {})",
            self.output.to_token_stream().to_string(),
            self.method.to_token_stream().to_string(),
            self.operand.to_token_stream().to_string(),
            self.args
                .iter()
                .map(|arg| arg.to_token_stream().to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl ToTokens for Unary {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let method = proc_macro2::Ident::new(&format!("_{}", self.method), self.method.span());
        let operand = proc_macro2::Ident::new(&format!("{}", self.operand), self.operand.span());
        let output = proc_macro2::Ident::new(&format!("{}", self.output), self.output.span());
        if self.args.is_empty() {
            tokens.extend(
                quote::quote!(
                let #output = #operand.#method();
            )
            );
        } else {
            let args = self.args.iter();
            tokens.extend(
                quote::quote!(
                let #output = #operand.#method(#(#args),*);
            )
            );
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Binary {
    pub(crate) method: syn::Ident,
    pub(crate) left: syn::Ident,
    pub(crate) right: syn::Ident,
    pub(crate) output: syn::Ident,
    pub(crate) kernel_type: KernelType,
}

impl std::fmt::Debug for Binary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} = {}({}, {})",
            {
                self.output.to_token_stream().to_string()
            },
            self.method.to_token_stream().to_string(),
            self.left.to_token_stream().to_string(),
            self.right.to_token_stream().to_string()
        )
    }
}

impl<'ast> ToTokens for Binary {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let method = proc_macro2::Ident::new(&format!("_{}", self.method), self.method.span());
        let left = proc_macro2::Ident::new(&format!("{}", self.left), self.left.span());
        let right = proc_macro2::Ident::new(&format!("{}", self.right), self.right.span());
        let output = proc_macro2::Ident::new(&format!("{}", self.output), self.output.span());
        tokens.extend(quote::quote!(
             let #output = #left.#method(#right);
        ));
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) enum Node {
    Unary(Unary),
    Binary(Binary),
    Input(syn::Ident),
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unary(unary, ..) => write!(f, "{:#?}", unary),
            Node::Binary(binary, ..) => write!(f, "{:#?}", binary),
            Node::Input(input) => write!(f, "input({})", input.to_string()),
        }
    }
}
